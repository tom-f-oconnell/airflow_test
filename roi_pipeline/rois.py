"""Turning spatial components into ROIs and extracting their time courses."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, sparse

from .datasets import Movie
from .factorize import downsample_block


@dataclass
class ROI:
    """One connected region claimed from a spatial component.

    Coordinates are in the (possibly downsampled) component grid.
    """

    id: int
    component: int
    n_voxels: int
    centroid: list[float]
    bbox: list[list[int]]
    peak_weight: float

    def to_dict(self) -> dict:
        return asdict(self)


def robust_zscore(a: np.ndarray) -> np.ndarray:
    """(a - median) / (1.4826 * MAD), falling back to the std for sparse maps."""
    med = float(np.median(a))
    scale = 1.4826 * float(np.median(np.abs(a - med)))
    if scale < 1e-12:
        scale = float(a.std())
    if scale < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - med) / scale).astype(np.float32)


def _describe(labels: np.ndarray, roi_id: int, component: int, spatial_map: np.ndarray) -> ROI | None:
    mask = labels == roi_id
    n = int(mask.sum())
    if n == 0:
        return None
    coords = np.nonzero(mask)
    centroid = [float(c.mean()) for c in coords]
    bbox = [[int(c.min()), int(c.max()) + 1] for c in coords]
    return ROI(id=roi_id, component=component, n_voxels=n, centroid=centroid, bbox=bbox,
               peak_weight=float(spatial_map[mask].max()))


def segment_components(
    spatial: np.ndarray,
    *,
    threshold_sigma: float = 3.0,
    min_voxels: int = 20,
    max_rois_per_component: int = 3,
) -> tuple[np.ndarray, list[ROI]]:
    """Threshold each spatial map, label connected regions, resolve overlaps.

    A voxel claimed by several components goes to the one with the larger
    z-scored weight. Returns a label volume (0 = background, i = ROI id) and
    the ROI descriptors, numbered consecutively from 1.
    """
    labels = np.zeros(spatial.shape[1:], np.int32)
    best = np.full(spatial.shape[1:], -np.inf, np.float32)
    owners: list[tuple[int, int]] = []  # provisional id -> component
    structure = ndimage.generate_binary_structure(3, 1)
    for c in range(spatial.shape[0]):
        z = robust_zscore(spatial[c])
        mask = z > threshold_sigma
        if int(mask.sum()) < min_voxels:
            continue
        lab, n = ndimage.label(mask, structure=structure)
        sizes = ndimage.sum_labels(mask, lab, index=np.arange(1, n + 1))
        for idx in np.argsort(sizes)[::-1][:max_rois_per_component]:
            if sizes[idx] < min_voxels:
                break
            claim = (lab == idx + 1) & (z > best)
            if int(claim.sum()) < min_voxels:
                continue
            provisional = len(owners) + 1
            labels[claim] = provisional
            best[claim] = z[claim]
            owners.append((provisional, c))
    # Later components may have stolen voxels; re-measure and renumber.
    final_labels = np.zeros_like(labels)
    rois: list[ROI] = []
    for provisional, component in owners:
        mask = labels == provisional
        if int(mask.sum()) < min_voxels:
            continue
        roi_id = len(rois) + 1
        final_labels[mask] = roi_id
        rois.append(_describe(final_labels, roi_id, component, spatial[component]))
    return final_labels, rois


def roi_traces(movie: Movie, labels: np.ndarray, *, downsample: int = 1, chunk_frames: int = 100) -> np.ndarray:
    """Mean raw signal per ROI per frame, streamed over time chunks.

    Returns (T, n_rois) float32; column j holds ROI id j + 1.
    """
    n = int(labels.max())
    traces = np.zeros((movie.n_frames, n), np.float32)
    if n == 0:
        return traces
    flat = labels.ravel()
    idx = np.nonzero(flat)[0]
    cols = flat[idx] - 1
    counts = np.bincount(cols, minlength=n).astype(np.float32)
    averaging = sparse.csr_matrix((1.0 / counts[cols], (idx, cols)), shape=(flat.size, n), dtype=np.float32)
    for t0, block in movie.iter_chunks(chunk_frames):
        part = downsample_block(block, downsample).reshape(len(block), -1)
        traces[t0:t0 + len(part)] = averaging.T.dot(part.T).T
    return traces


def dff(traces: np.ndarray, baseline_percentile: float = 10.0, eps: float = 1e-6) -> np.ndarray:
    """(F - F0) / F0 with F0 the per-ROI low percentile over time."""
    f0 = np.percentile(traces, baseline_percentile, axis=0)
    return ((traces - f0) / np.maximum(f0, eps)).astype(np.float32)
