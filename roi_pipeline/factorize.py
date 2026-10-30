"""Matrix factorization of a movie into spatial and temporal components.

The movie (T, Z, Y, X) is treated as a matrix X with one row per frame and one
column per voxel. Two families are offered:

* ``nmf``: non-negative matrix factorization of the mean-subtracted, clipped
  movie (W: T x k temporal courses, H: k x P spatial maps). Components are
  sparse and positive, which maps naturally onto cells with calcium
  transients.
* ``pca_ica``: truncated SVD followed by spatial ICA on the leading components
  (Mukamel et al. 2009 style). Better at splitting overlapping cells; the sign
  ambiguity is resolved by making each spatial map positively skewed.

Each has an in-memory solver and a streaming one (time chunks fed to sklearn's
MiniBatchNMF / IncrementalPCA). :mod:`roi_pipeline.memory` chooses between
them from the data node's memory cap.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sstats

from . import memory
from .datasets import Movie

log = logging.getLogger(__name__)

METHODS = ("nmf", "pca_ica")
MODES = ("in_memory", "chunked")


@dataclass
class Factorization:
    """Spatial maps (k, Z, Y', X'), temporal courses (T, k) and run statistics."""

    spatial: np.ndarray
    temporal: np.ndarray
    method: str
    mode: str
    downsample: int
    stats: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return self.spatial.shape[0]


def downsample_block(block: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a (T, Z, Y, X) block over factor x factor pixels in Y and X."""
    factor = int(factor)
    if factor <= 1:
        return block
    n_t, n_z, n_y, n_x = block.shape
    y_c, x_c = (n_y // factor) * factor, (n_x // factor) * factor
    view = block[:, :, :y_c, :x_c].reshape(n_t, n_z, y_c // factor, factor, x_c // factor, factor)
    return view.mean(axis=(3, 5), dtype=np.float32)


def downsampled_shape(frame_shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    n_z, n_y, n_x = frame_shape
    factor = max(1, int(factor))
    return n_z, n_y // factor, n_x // factor


def _matrix(block: np.ndarray) -> np.ndarray:
    return block.reshape(block.shape[0], -1)


def load_matrix(movie: Movie, downsample: int, chunk_frames: int) -> np.ndarray:
    """Whole movie as float32 (T, P), filled chunk by chunk to avoid a double copy."""
    n_voxels = int(np.prod(downsampled_shape(movie.frame_shape, downsample)))
    matrix = np.empty((movie.n_frames, n_voxels), np.float32)
    for t0, block in movie.iter_chunks(chunk_frames):
        matrix[t0:t0 + len(block)] = _matrix(downsample_block(block, downsample))
    return matrix


def streaming_mean(movie: Movie, downsample: int, chunk_frames: int) -> np.ndarray:
    total = None
    for _, block in movie.iter_chunks(chunk_frames):
        part = _matrix(downsample_block(block, downsample)).sum(axis=0, dtype=np.float64)
        total = part if total is None else total + part
    return (total / movie.n_frames).astype(np.float32)


def _positive_skew(spatial: np.ndarray, temporal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip components whose spatial map is negatively skewed (ICA sign ambiguity)."""
    for c in range(spatial.shape[0]):
        if sstats.skew(spatial[c]) < 0:
            spatial[c] *= -1
            temporal[:, c] *= -1
    return spatial, temporal


def _spatial_ica(components: np.ndarray, scale: np.ndarray, seed: int) -> np.ndarray:
    """Independent spatial maps from scaled principal components (k, P)."""
    from sklearn.decomposition import FastICA

    k = components.shape[0]
    pcs = (components * scale[:, None]).T.astype(np.float64)  # (P, k)
    ica = FastICA(n_components=k, whiten="unit-variance", random_state=seed, max_iter=500, tol=1e-3)
    sources = ica.fit_transform(pcs)  # (P, k)
    return sources.T.astype(np.float32)


def _projection_operator(spatial: np.ndarray) -> np.ndarray:
    """A such that W = X @ A.T is the least-squares temporal course for maps H."""
    gram = spatial.astype(np.float64) @ spatial.astype(np.float64).T
    gram.flat[:: gram.shape[0] + 1] += 1e-6 * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram, spatial.astype(np.float64)).astype(np.float32)


def _nmf_in_memory(matrix: np.ndarray, k: int, seed: int, max_iter: int):
    from sklearn.decomposition import NMF

    matrix -= matrix.mean(axis=0)
    np.maximum(matrix, 0, out=matrix)
    model = NMF(n_components=k, init="nndsvda", max_iter=max_iter, tol=1e-4, random_state=seed)
    temporal = model.fit_transform(matrix)
    spatial = model.components_
    stats = {"reconstruction_err": float(model.reconstruction_err_), "n_iter": int(model.n_iter_)}
    return spatial.astype(np.float32), temporal.astype(np.float32), stats


def _pca_ica_in_memory(matrix: np.ndarray, k: int, seed: int):
    from sklearn.utils.extmath import randomized_svd

    matrix -= matrix.mean(axis=0)
    _, singular, components = randomized_svd(matrix, n_components=k, n_iter=4, random_state=seed)
    total_var = float(np.linalg.norm(matrix) ** 2)
    spatial = _spatial_ica(components, singular, seed)
    temporal = matrix @ _projection_operator(spatial).T
    stats = {"explained_variance_ratio": float((singular ** 2).sum() / max(total_var, 1e-12))}
    return spatial, temporal.astype(np.float32), stats


def _nmf_chunked(movie: Movie, k: int, downsample: int, chunk_frames: int, seed: int, epochs: int):
    from sklearn.decomposition import MiniBatchNMF

    mean = streaming_mean(movie, downsample, chunk_frames)
    model = MiniBatchNMF(n_components=k, batch_size=max(1, min(1024, chunk_frames)), random_state=seed)

    def chunks():
        for t0, block in movie.iter_chunks(chunk_frames):
            part = _matrix(downsample_block(block, downsample))
            part -= mean
            np.maximum(part, 0, out=part)
            yield t0, part

    for epoch in range(epochs):
        for t0, part in chunks():
            model.partial_fit(part)
        log.info("MiniBatchNMF epoch %d/%d done", epoch + 1, epochs)
    temporal = np.empty((movie.n_frames, k), np.float32)
    for t0, part in chunks():
        temporal[t0:t0 + len(part)] = model.transform(part)
    return model.components_.astype(np.float32), temporal, {"epochs": epochs}


def _pca_ica_chunked(movie: Movie, k: int, downsample: int, chunk_frames: int, seed: int):
    from sklearn.decomposition import IncrementalPCA

    model = IncrementalPCA(n_components=k)
    for _, block in movie.iter_chunks(chunk_frames):
        model.partial_fit(_matrix(downsample_block(block, downsample)))
    scale = np.sqrt(model.explained_variance_ * max(model.n_samples_seen_ - 1, 1))
    spatial = _spatial_ica(model.components_, scale, seed)
    operator = _projection_operator(spatial)
    temporal = np.empty((movie.n_frames, k), np.float32)
    mean = model.mean_.astype(np.float32)
    for t0, block in movie.iter_chunks(chunk_frames):
        part = _matrix(downsample_block(block, downsample)) - mean
        temporal[t0:t0 + len(part)] = part @ operator.T
    stats = {"explained_variance_ratio": float(model.explained_variance_ratio_.sum())}
    return spatial, temporal, stats


def run_factorization(
    movie: Movie,
    *,
    method: str = "nmf",
    n_components: int = 20,
    mode: str = "in_memory",
    downsample: int = 1,
    chunk_frames: int = 100,
    seed: int = 0,
    max_iter: int = 300,
    chunked_epochs: int = 2,
) -> Factorization:
    """Factorize ``movie`` and return spatial maps at the downsampled resolution.

    Args:
        movie: Source movie.
        method: ``nmf`` or ``pca_ica``.
        n_components: Rank k (clamped to min(T, P)).
        mode: ``in_memory`` loads the full matrix; ``chunked`` streams it.
        downsample: Spatial block-mean factor applied to Y and X.
        chunk_frames: Frames per read; also the mini-batch size when chunked.
        seed: Random state for the solvers.
        max_iter: NMF iteration cap (in-memory).
        chunked_epochs: Passes over the data for MiniBatchNMF.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    frame_shape = downsampled_shape(movie.frame_shape, downsample)
    n_voxels = int(np.prod(frame_shape))
    k = int(min(n_components, movie.n_frames, n_voxels))
    if mode == "chunked":
        k = min(k, max(1, min(chunk_frames, movie.n_frames)))
    if k != n_components:
        log.warning("n_components reduced from %d to %d to fit the data", n_components, k)
    log.info("factorize %s: method=%s mode=%s k=%d downsample=%d matrix=(%d, %d)",
             movie.name, method, mode, k, downsample, movie.n_frames, n_voxels)
    started = time.time()
    rss_before = memory.current_rss_bytes()
    if mode == "in_memory":
        matrix = load_matrix(movie, downsample, chunk_frames)
        if method == "nmf":
            spatial, temporal, stats = _nmf_in_memory(matrix, k, seed, max_iter)
        else:
            spatial, temporal, stats = _pca_ica_in_memory(matrix, k, seed)
        del matrix
    elif method == "nmf":
        spatial, temporal, stats = _nmf_chunked(movie, k, downsample, chunk_frames, seed, chunked_epochs)
    else:
        spatial, temporal, stats = _pca_ica_chunked(movie, k, downsample, chunk_frames, seed)
    spatial, temporal = _positive_skew(spatial, temporal)
    stats.update({
        "seconds": time.time() - started,
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": memory.peak_rss_bytes(),
        "n_components": k,
        "matrix_shape": [movie.n_frames, n_voxels],
    })
    return Factorization(spatial=spatial.reshape(k, *frame_shape), temporal=temporal,
                         method=method, mode=mode, downsample=int(downsample), stats=stats)
