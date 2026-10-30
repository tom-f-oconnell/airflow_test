"""Synthetic volumetric calcium-imaging movies with known ROIs.

The generator streams frames into a zarr group chunk by chunk, so a movie
larger than the data node's memory cap can be produced on that node (that is
the point of the memory-cap tests). Ground truth (ROI centres and sizes, the
stimuli each ROI responds to) and the stimulus schedule go into the zarr attrs
so tests and the dashboard can score a run against them.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

BASELINE = 1000.0  # mean background in uint16 counts
SIGNAL_SCALE = 400.0  # counts per unit of response amplitude


def calcium_kernel(n: int, tau_rise: float = 1.5, tau_decay: float = 10.0) -> np.ndarray:
    """Difference-of-exponentials transient, peak normalised to 1 (frames)."""
    t = np.arange(n, dtype=np.float64)
    k = (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    return (k / k.max()).astype(np.float32)


def make_stimulus_schedule(n_frames: int, n_stimuli: int, n_trials: int, stim_frames: int,
                           pre_frames: int, post_frames: int, rng: np.random.Generator) -> dict:
    """Block design: every block presents each stimulus once in random order."""
    period = pre_frames + stim_frames + post_frames
    ids = [str(s) for s in range(n_stimuli)]
    onsets = {s: [] for s in ids}
    t = pre_frames
    for _ in range(n_trials):
        for s in rng.permutation(n_stimuli):
            if t + stim_frames + post_frames > n_frames:
                break
            onsets[str(s)].append(int(t))
            t += period
    return {"ids": ids, "onsets": onsets, "duration_frames": int(stim_frames),
            "pre_frames": int(pre_frames), "post_frames": int(post_frames)}


def gaussian_patch(center, sigma, shape, n_sigma: float = 3.0):
    """Gaussian blob restricted to its bounding box: returns (slices, patch)."""
    slices, grids = [], []
    for c, s, n in zip(center, sigma, shape):
        lo = max(0, int(math.floor(c - n_sigma * s)))
        hi = min(n, int(math.ceil(c + n_sigma * s)) + 1)
        slices.append(slice(lo, hi))
        grids.append(np.arange(lo, hi, dtype=np.float32))
    zz, yy, xx = np.meshgrid(*grids, indexing="ij")
    patch = np.exp(-0.5 * (((zz - center[0]) / sigma[0]) ** 2
                           + ((yy - center[1]) / sigma[1]) ** 2
                           + ((xx - center[2]) / sigma[2]) ** 2))
    return tuple(slices), patch.astype(np.float32)


def generate_synthetic_movie(
    path: Path | str,
    *,
    n_frames: int = 300,
    n_z: int = 8,
    height: int = 64,
    width: int = 64,
    n_rois: int = 12,
    n_stimuli: int = 3,
    n_trials: int = 4,
    stim_frames: int = 8,
    pre_frames: int = 15,
    post_frames: int = 25,
    response_prob: float = 0.5,
    noise_sd: float = 25.0,
    frame_rate_hz: float = 5.0,
    seed: int = 0,
    chunk_frames: int = 20,
) -> dict:
    """Write a synthetic movie to ``path`` (a zarr group) and return its attrs.

    Each ROI is a 3-D Gaussian blob with a calcium-like trace made of
    stimulus-locked responses (to a random subset of stimuli) plus sparse
    spontaneous events, on a slowly bleaching textured background with
    Gaussian noise. Stored as uint16 like camera data.
    """
    import zarr

    rng = np.random.default_rng(seed)
    shape = (n_z, height, width)
    schedule = make_stimulus_schedule(n_frames, n_stimuli, n_trials, stim_frames, pre_frames, post_frames, rng)
    kernel = calcium_kernel(min(n_frames, 60))

    rois = []
    for i in range(n_rois):
        center = (float(rng.uniform(0.5, max(0.5, n_z - 0.5))),
                  float(rng.uniform(6, height - 6)),
                  float(rng.uniform(6, width - 6)))
        sigma = (float(rng.uniform(0.7, 1.4)), float(rng.uniform(1.8, 3.2)), float(rng.uniform(1.8, 3.2)))
        amplitude = float(rng.uniform(0.6, 1.5))
        responsive = [s for s in schedule["ids"] if rng.random() < response_prob]
        drive = np.zeros(n_frames, np.float32)
        for s in responsive:
            gain = rng.uniform(0.6, 1.2)
            for onset in schedule["onsets"][s]:
                drive[onset:onset + stim_frames] += gain
        for t in rng.integers(0, n_frames, rng.poisson(n_frames / 150)):
            drive[t] += rng.uniform(0.3, 0.8)
        trace = (np.convolve(drive, kernel)[:n_frames] * amplitude).astype(np.float32)
        slices, patch = gaussian_patch(center, sigma, shape)
        rois.append({"id": i, "center": list(center), "sigma": list(sigma), "amplitude": amplitude,
                     "responsive_stimuli": responsive, "_trace": trace, "_slices": slices, "_patch": patch})

    frames_t = np.arange(n_frames, dtype=np.float32)
    drift = (1 - 0.05 * frames_t / max(n_frames - 1, 1)
             + 0.01 * np.sin(2 * np.pi * frames_t / max(n_frames / 3, 1))).astype(np.float32)
    texture = ndimage.gaussian_filter(rng.random(shape), sigma=(0.5, 3, 3))
    texture = (texture - texture.min()) / max(np.ptp(texture), 1e-6)
    background = (BASELINE * (0.8 + 0.4 * texture)).astype(np.float32)

    path = Path(path)
    group = zarr.open_group(str(path), mode="w")
    movie = group.create_array("movie", shape=(n_frames, *shape),
                               chunks=(max(1, min(chunk_frames, n_frames)), *shape), dtype="uint16")
    for t0 in range(0, n_frames, chunk_frames):
        t1 = min(t0 + chunk_frames, n_frames)
        block = background[None] * drift[t0:t1, None, None, None]
        for roi in rois:
            block[(slice(None), *roi["_slices"])] += (
                SIGNAL_SCALE * roi["_trace"][t0:t1, None, None, None] * roi["_patch"][None])
        block += rng.normal(0, noise_sd, block.shape).astype(np.float32)
        movie[t0:t1] = np.clip(block, 0, 65535).astype(np.uint16)
        log.info("synthetic %s: wrote frames %d-%d of %d", path.name, t0, t1, n_frames)

    attrs = {
        "format": "roi-pipeline-movie/1",
        "axes": "tzyx",
        "frame_rate_hz": float(frame_rate_hz),
        "stimuli": schedule,
        "ground_truth": {"rois": [{k: v for k, v in roi.items() if not k.startswith("_")} for roi in rois]},
        "generator": {"n_frames": n_frames, "n_z": n_z, "height": height, "width": width, "n_rois": n_rois,
                      "n_stimuli": n_stimuli, "n_trials": n_trials, "stim_frames": stim_frames,
                      "pre_frames": pre_frames, "post_frames": post_frames, "response_prob": response_prob,
                      "noise_sd": noise_sd, "seed": seed, "chunk_frames": chunk_frames},
    }
    group.attrs.update(attrs)
    return attrs
