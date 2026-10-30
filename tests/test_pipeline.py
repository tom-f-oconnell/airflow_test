import json
from pathlib import Path

import numpy as np
import pytest

from roi_pipeline import classify, datasets, factorize, memory, preview, rois, synthetic

DAGS = Path(__file__).resolve().parents[1] / "dags"


def recall(movie, found, downsample=1, tol=3.0):
    """Fraction of planted ROIs with an extracted centroid within ``tol`` voxels."""
    truth = movie.attrs["ground_truth"]["rois"]
    if not found:
        return 0.0
    centroids = np.array([r.centroid for r in found])
    hits = 0
    for roi in truth:
        z, y, x = roi["center"]
        hits += np.linalg.norm(centroids - [z, y / downsample, x / downsample], axis=1).min() <= tol
    return hits / len(truth)


def test_parse_and_format_size():
    assert memory.parse_size("512m") == 512 * 2**20
    assert memory.parse_size("8g") == 8 * 2**30
    assert memory.parse_size("1.5G") == int(1.5 * 2**30)
    assert memory.parse_size("1073741824") == 2**30
    assert memory.format_size(2**30) == "1.00 GiB"
    assert memory.format_size(None) == "unlimited"
    with pytest.raises(ValueError):
        memory.parse_size("lots")


def test_plan_memory_modes():
    big = memory.plan_memory(1000, 4_000_000, cap_bytes=memory.parse_size("4g"), chunk_frames=20)
    assert big.mode == "chunked" and not big.fits_in_memory and big.fits_chunked
    small = memory.plan_memory(100, 10_000, cap_bytes=memory.parse_size("4g"))
    assert small.mode == "in_memory" and small.fits_in_memory
    forced = memory.plan_memory(1000, 4_000_000, cap_bytes=memory.parse_size("4g"), requested_mode="in_memory")
    assert forced.mode == "in_memory" and not forced.fits_in_memory
    assert memory.plan_memory(10, 10, cap_bytes=2**30, chunk_frames=100).chunk_frames == 10
    with pytest.raises(ValueError):
        memory.plan_memory(10, 10, requested_mode="sideways", cap_bytes=2**30)
    assert memory.memory_cap_bytes() > 0
    assert json.dumps(big.to_dict())


def test_open_movie_and_listing(data_root, small_movie_path):
    movie = datasets.open_movie(small_movie_path)
    assert movie.shape == (160, 4, 32, 32)
    assert movie.dtype == np.uint16
    assert movie.read(0, 5).shape == (5, 4, 32, 32) and movie.read(0, 5).dtype == np.float32
    assert movie.sample_frames(4).shape == (4, 4, 32, 32)
    assert movie.stimulus_schedule()["ids"] == ["0", "1"]
    assert len(movie.attrs["ground_truth"]["rois"]) == 5
    listed = datasets.list_datasets()
    assert [d["name"] for d in listed] == ["synthetic_test"]
    assert listed[0]["has_stimuli"] and listed[0]["size_bytes"] == movie.nbytes
    assert datasets.resolve_dataset("synthetic_test") == small_movie_path
    with pytest.raises(FileNotFoundError):
        datasets.resolve_dataset("nope")
    assert datasets.safe_run_id("manual__2026-10-29T12:00:00+00:00") == "manual__2026-10-29T12_00_00_00_00"


def test_npy_and_tiff_sources(tmp_path):
    arr = (np.random.default_rng(0).random((6, 3, 8, 8)) * 100).astype(np.uint16)
    np.save(tmp_path / "m.npy", arr)
    (tmp_path / "m.json").write_text(json.dumps({"frame_rate_hz": 2.0}))
    npy = datasets.open_movie(tmp_path / "m.npy")
    assert npy.shape == (6, 3, 8, 8) and npy.attrs["frame_rate_hz"] == 2.0
    np.testing.assert_array_equal(npy.read(2, 4), arr[2:4])
    import tifffile

    tifffile.imwrite(tmp_path / "m.tif", arr, photometric="minisblack")
    tif = datasets.open_movie(tmp_path / "m.tif")
    assert tif.shape == (6, 3, 8, 8)
    np.testing.assert_array_equal(tif.read(1, 3), arr[1:3])
    flat = arr[:, 0]
    np.save(tmp_path / "flat.npy", flat)
    assert datasets.open_movie(tmp_path / "flat.npy").shape == (6, 1, 8, 8)


def test_downsample_block():
    block = np.arange(2 * 1 * 4 * 6, dtype=np.float32).reshape(2, 1, 4, 6)
    ds = factorize.downsample_block(block, 2)
    assert ds.shape == (2, 1, 2, 3)
    assert ds[0, 0, 0, 0] == pytest.approx(block[0, 0, :2, :2].mean())
    assert factorize.downsampled_shape((4, 33, 32), 2) == (4, 16, 16)


@pytest.mark.parametrize("method,mode", [("nmf", "in_memory"), ("nmf", "chunked"),
                                         ("pca_ica", "in_memory"), ("pca_ica", "chunked")])
def test_factorization_recovers_planted_rois(small_movie_path, method, mode):
    movie = datasets.open_movie(small_movie_path)
    result = factorize.run_factorization(movie, method=method, n_components=8, mode=mode, chunk_frames=40, seed=0)
    assert result.spatial.shape == (8, 4, 32, 32)
    assert result.temporal.shape == (160, 8)
    assert result.stats["seconds"] >= 0 and result.stats["peak_rss_bytes"] > 0
    labels, found = rois.segment_components(result.spatial, threshold_sigma=3.0, min_voxels=8)
    assert labels.shape == (4, 32, 32)
    assert len(found) >= 3
    assert recall(movie, found) >= 0.6, [r.centroid for r in found]
    for roi in found:
        assert (labels == roi.id).sum() == roi.n_voxels


def test_downsampled_factorization(small_movie_path):
    movie = datasets.open_movie(small_movie_path)
    result = factorize.run_factorization(movie, method="nmf", n_components=8, mode="in_memory", downsample=2)
    assert result.spatial.shape == (8, 4, 16, 16)
    labels, found = rois.segment_components(result.spatial, threshold_sigma=2.5, min_voxels=3)
    traces = rois.roi_traces(movie, labels, downsample=2, chunk_frames=50)
    assert traces.shape == (160, len(found))
    assert recall(movie, found, downsample=2, tol=2.5) >= 0.6


def test_traces_and_classification(small_movie_path, data_root):
    movie = datasets.open_movie(small_movie_path)
    result = factorize.run_factorization(movie, method="nmf", n_components=8, mode="in_memory")
    labels, found = rois.segment_components(result.spatial, threshold_sigma=3.0, min_voxels=8)
    traces = rois.roi_traces(movie, labels, chunk_frames=64)
    assert traces.shape == (160, len(found))
    # Streaming mean per ROI must equal the direct computation.
    direct = movie.read(0, 160).reshape(160, -1)[:, labels.ravel() == found[0].id].mean(axis=1)
    np.testing.assert_allclose(traces[:, 0], direct, rtol=1e-4)
    dff = rois.dff(traces)
    assert dff.shape == traces.shape and np.all(np.isfinite(dff))

    out = classify.classify_rois([r.to_dict() for r in found], dff, movie.stimulus_schedule(),
                                 n_response_classes=2, model_path=data_root / "models" / "none.joblib")
    assert out["quality_source"] == "heuristic"
    assert len(out["rois"]) == len(found)
    assert sum(out["quality_counts"].values()) == len(found)
    assert out["grouping"] == "stimulus_tuning"
    assert out["stimuli"]["ids"] == ["0", "1"]
    assert set(r["quality"]["label"] for r in out["rois"]) <= set(classify.QUALITY_LABELS)
    json.dumps(out)

    # Responsiveness against the planted truth, over matched ROIs.
    truth = movie.attrs["ground_truth"]["rois"]
    agree = total = 0
    for roi, entry in zip(found, out["rois"]):
        dist = [np.linalg.norm(np.array(roi.centroid) - t["center"]) for t in truth]
        if min(dist) > 3:
            continue
        planted = set(truth[int(np.argmin(dist))]["responsive_stimuli"])
        for stim in ("0", "1"):
            total += 1
            agree += (stim in planted) == (stim in entry["responsive_stimuli"])
    assert total >= 3 and agree / total >= 0.7

    without = classify.classify_rois([r.to_dict() for r in found], dff, None, n_response_classes=2)
    assert without["grouping"] == "trace_similarity" and "stimuli" not in without


def test_trained_quality_model(tmp_path):
    rng = np.random.default_rng(0)
    feats = [{n: float(v) for n, v in zip(classify.FEATURE_NAMES, rng.random(len(classify.FEATURE_NAMES)))}
             for _ in range(20)]
    labels = ["cell" if f["snr"] > 0.5 else "artifact" for f in feats]
    path = tmp_path / "roi_quality.joblib"
    classify.train_quality_model(feats, labels, path)
    model = classify.load_quality_model(path)
    out = classify.classify_quality(feats, model)
    assert out[0]["source"] == "model" and out[0]["label"] in ("cell", "artifact")
    assert classify.load_quality_model(tmp_path / "missing.joblib") is None


def test_preview_png(small_movie_path):
    movie = datasets.open_movie(small_movie_path)
    png = preview.projection_png(movie, downsample=1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    labels = np.zeros((4, 32, 32), np.int32)
    labels[1, 5:10, 5:10] = 1
    assert preview.projection_png(movie, labels=labels)[:8] == b"\x89PNG\r\n\x1a\n"
    tiles, cols = preview.montage(np.zeros((5, 4, 6)))
    assert tiles.shape == (8, 18) and cols == 3


def test_synthetic_schedule_fits():
    rng = np.random.default_rng(0)
    sched = synthetic.make_stimulus_schedule(220, 3, 2, 5, 10, 20, rng)
    onsets = sorted(t for s in sched["ids"] for t in sched["onsets"][s])
    assert len(onsets) == 6 and onsets[0] == 10 and all(b - a == 35 for a, b in zip(onsets, onsets[1:]))


def test_results_bookkeeping(data_root):
    run = datasets.run_dir("synthetic_test", "manual__2026-10-29T00:00:00+00:00", create=True)
    datasets.write_json(run / "plan.json", {"state": "running", "x": np.float32(1.5)})
    datasets.write_json(run / "summary.json", {"state": "success"})
    listed = datasets.list_results("synthetic_test")
    assert listed[0]["run"] == run.name and listed[0]["plan"]["x"] == 1.5
    assert listed[0]["summary"]["state"] == "success" and not listed[0]["has_classification"]


@pytest.mark.parametrize("dag_file", sorted(DAGS.glob("*.py")), ids=lambda p: p.name)
def test_dag_files_compile(dag_file):
    compile(dag_file.read_text(), str(dag_file), "exec")
