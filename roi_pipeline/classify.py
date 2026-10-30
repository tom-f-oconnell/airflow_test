"""Classification of extracted ROIs.

Two independent questions are answered here:

1. ROI quality ("is this a cell?"): a feature vector per ROI (shape and
   trace statistics) scored by a rule-based classifier, or by a trained
   sklearn model if one is supplied (``models/roi_quality.joblib`` on the
   data volume, built with :func:`train_quality_model` from hand labels).
2. Response type: with the stimulus schedule stored on the dataset, which
   stimuli each ROI responds to (trial-averaged response z-scored against
   the pre-stimulus baseline) and a k-means grouping of the responsive ROIs
   by their tuning across stimuli. Without a schedule, ROIs are grouped by
   trace similarity (co-active ensembles) instead.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import stats as sstats

log = logging.getLogger(__name__)

FEATURE_NAMES = ("n_voxels", "compactness", "aspect_ratio", "snr", "skewness", "lag1_autocorr", "max_dff")
QUALITY_LABELS = ("cell", "uncertain", "artifact")


def roi_features(roi: dict, trace_dff: np.ndarray) -> dict:
    """Shape and trace statistics for one ROI (see ``FEATURE_NAMES``)."""
    bbox = np.asarray(roi["bbox"])
    extent = np.maximum(bbox[:, 1] - bbox[:, 0], 1)
    noise = 1.4826 * float(np.median(np.abs(np.diff(trace_dff)))) / np.sqrt(2) + 1e-9
    lag1 = float(np.corrcoef(trace_dff[:-1], trace_dff[1:])[0, 1]) if len(trace_dff) > 2 else 0.0
    return {
        "n_voxels": float(roi["n_voxels"]),
        "compactness": float(roi["n_voxels"] / np.prod(extent)),
        "aspect_ratio": float(extent[1:].max() / extent[1:].min()),
        "snr": float(trace_dff.max() / noise),
        "skewness": float(sstats.skew(trace_dff)),
        "lag1_autocorr": 0.0 if np.isnan(lag1) else lag1,
        "max_dff": float(trace_dff.max()),
    }


def feature_matrix(features: list[dict]) -> np.ndarray:
    return np.array([[f[name] for name in FEATURE_NAMES] for f in features], dtype=np.float64)


def heuristic_quality(f: dict) -> tuple[str, float]:
    """Score in [0, 1] from rules that describe a calcium-imaging cell body.

    Transients make the dF/F trace positively skewed and autocorrelated; a
    compact, roughly isotropic blob fills a sensible fraction of its box.
    """
    score = 0.0
    score += 0.4 if f["snr"] >= 8 else 0.3 if f["snr"] >= 4 else 0.1 if f["snr"] >= 2.5 else 0.0
    score += 0.2 if f["skewness"] > 0.5 else 0.1 if f["skewness"] > 0 else 0.0
    score += 0.15 if f["lag1_autocorr"] > 0.3 else 0.05 if f["lag1_autocorr"] > 0.1 else 0.0
    score += 0.15 if 0.2 <= f["compactness"] <= 0.95 else 0.0
    score += 0.1 if f["aspect_ratio"] < 3 else 0.0
    label = "cell" if score >= 0.6 else "artifact" if score <= 0.35 else "uncertain"
    return label, round(score, 3)


def load_quality_model(path: Path | str | None):
    """Trained sklearn classifier (joblib) or None when absent."""
    if not path or not Path(path).exists():
        return None
    import joblib

    return joblib.load(path)


def train_quality_model(features: list[dict], labels: list[str], path: Path | str):
    """Fit a random forest on hand-labelled ROIs and save it with joblib."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
    model.fit(feature_matrix(features), labels)
    joblib.dump(model, path)
    return model


def classify_quality(features: list[dict], model=None) -> list[dict]:
    if model is None:
        return [dict(zip(("label", "score"), heuristic_quality(f)), source="heuristic") for f in features]
    matrix = feature_matrix(features)
    labels = model.predict(matrix)
    scores = model.predict_proba(matrix).max(axis=1) if hasattr(model, "predict_proba") else np.ones(len(labels))
    return [{"label": str(lab), "score": round(float(s), 3), "source": "model"} for lab, s in zip(labels, scores)]


def stimulus_responses(traces_dff: np.ndarray, schedule: dict, *, baseline_frames: int | None = None,
                       response_frames: int | None = None, z_threshold: float = 2.5) -> dict:
    """Trial-averaged response of every ROI to every stimulus.

    For each trial the mean over the response window minus the mean over the
    preceding baseline window is taken; its average over trials is divided by
    the standard error expected from the baseline noise, giving a z-score.
    """
    n_frames, n_rois = traces_dff.shape
    ids = list(schedule["ids"])
    pre = int(baseline_frames or schedule.get("pre_frames") or 10)
    resp = int(response_frames or (schedule["duration_frames"] + 5))
    z = np.zeros((n_rois, len(ids)), np.float32)
    timecourses = np.zeros((len(ids), pre + resp, n_rois), np.float32)
    n_trials = []
    for s, stim in enumerate(ids):
        diffs, baselines, windows = [], [], []
        for onset in schedule["onsets"][stim]:
            if onset - pre < 0 or onset + resp > n_frames:
                continue
            base = traces_dff[onset - pre:onset]
            window = traces_dff[onset - pre:onset + resp]
            diffs.append(traces_dff[onset:onset + resp].mean(axis=0) - base.mean(axis=0))
            baselines.append(base)
            windows.append(window)
        n_trials.append(len(diffs))
        if not diffs:
            continue
        diffs = np.array(diffs)
        sigma = np.concatenate(baselines).std(axis=0) + 1e-6
        stderr = sigma * np.sqrt(1 / resp + 1 / pre) / np.sqrt(len(diffs))
        z[:, s] = diffs.mean(axis=0) / stderr
        timecourses[s] = np.mean(windows, axis=0)
    return {"ids": ids, "z": z, "responsive": z > z_threshold, "timecourses": timecourses,
            "n_trials": n_trials, "baseline_frames": pre, "response_frames": resp,
            "z_threshold": z_threshold}


def cluster_tuning(z: np.ndarray, responsive: np.ndarray, n_classes: int, seed: int = 0) -> np.ndarray:
    """K-means on the normalised tuning vectors of responsive ROIs (-1 = silent)."""
    from sklearn.cluster import KMeans

    classes = np.full(z.shape[0], -1, np.int32)
    rows = np.nonzero(responsive.any(axis=1))[0]
    if len(rows) == 0:
        return classes
    k = max(1, min(int(n_classes), len(rows)))
    tuning = np.maximum(z[rows], 0)
    tuning /= np.linalg.norm(tuning, axis=1, keepdims=True) + 1e-9
    classes[rows] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(tuning)
    return classes


def cluster_by_similarity(traces_dff: np.ndarray, n_classes: int, seed: int = 0) -> np.ndarray:
    """K-means on z-scored traces: groups ROIs that are active together."""
    from sklearn.cluster import KMeans

    n_rois = traces_dff.shape[1]
    if n_rois == 0:
        return np.zeros(0, np.int32)
    k = max(1, min(int(n_classes), n_rois))
    normed = (traces_dff - traces_dff.mean(axis=0)) / (traces_dff.std(axis=0) + 1e-9)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(normed.T).astype(np.int32)


def classify_rois(rois: list[dict], traces_dff: np.ndarray, schedule: dict | None = None, *,
                  n_response_classes: int = 3, z_threshold: float = 2.5, model_path=None, seed: int = 0) -> dict:
    """Quality and response classification for every ROI; JSON-serialisable."""
    features = [roi_features(roi, traces_dff[:, i]) for i, roi in enumerate(rois)]
    model = load_quality_model(model_path)
    quality = classify_quality(features, model)
    responses = stimulus_responses(traces_dff, schedule, z_threshold=z_threshold) if schedule else None
    if responses is not None:
        classes = cluster_tuning(responses["z"], responses["responsive"], n_response_classes, seed)
        grouping = "stimulus_tuning"
    else:
        classes = cluster_by_similarity(traces_dff, n_response_classes, seed)
        grouping = "trace_similarity"
    out_rois = []
    for i, roi in enumerate(rois):
        entry = {"id": roi["id"], "features": features[i], "quality": quality[i],
                 "response_class": int(classes[i]) if i < len(classes) else -1}
        if responses is not None:
            entry["response_z"] = {stim: float(responses["z"][i, s]) for s, stim in enumerate(responses["ids"])}
            entry["responsive_stimuli"] = [stim for s, stim in enumerate(responses["ids"]) if responses["responsive"][i, s]]
        out_rois.append(entry)
    counts = {label: sum(q["label"] == label for q in quality) for label in QUALITY_LABELS}
    result = {
        "feature_names": list(FEATURE_NAMES),
        "quality_source": "model" if model is not None else "heuristic",
        "quality_counts": counts,
        "grouping": grouping,
        "n_response_classes": int(n_response_classes),
        "rois": out_rois,
    }
    if responses is not None:
        result["stimuli"] = {
            "ids": responses["ids"], "n_trials": responses["n_trials"],
            "baseline_frames": responses["baseline_frames"], "response_frames": responses["response_frames"],
            "z_threshold": z_threshold,
            "n_responsive": int(responses["responsive"].any(axis=1).sum()),
            # (n_stim, window, n_rois) trial-averaged dF/F, for the dashboard.
            "timecourses": responses["timecourses"].round(4).tolist(),
        }
    return result
