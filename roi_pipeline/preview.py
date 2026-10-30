"""Rendering helpers for the dashboard: projections, ROI overlays, PNG bytes."""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from .datasets import Movie
from .factorize import downsample_block

# Categorical slots of the dashboard palette (light mode), cycled over ROI ids.
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0),
           (232, 123, 164), (0, 131, 0), (98, 80, 214), (227, 73, 72)]


def max_projection(movie: Movie, *, n_frames: int = 12, downsample: int = 1) -> np.ndarray:
    """Max over ``n_frames`` evenly spaced frames: (Z, Y', X') float32."""
    return downsample_block(movie.sample_frames(n_frames), downsample).max(axis=0)


def to_uint8(volume: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(volume, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((volume - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def montage(volume: np.ndarray) -> tuple[np.ndarray, int]:
    """Tile the Z planes of (Z, Y, X) into a near-square 2-D grid."""
    n_z, n_y, n_x = volume.shape
    cols = int(math.ceil(math.sqrt(n_z)))
    rows = int(math.ceil(n_z / cols))
    out = np.zeros((rows * n_y, cols * n_x), volume.dtype)
    for z in range(n_z):
        r, c = divmod(z, cols)
        out[r * n_y:(r + 1) * n_y, c * n_x:(c + 1) * n_x] = volume[z]
    return out, cols


def overlay_rgb(gray_u8: np.ndarray, labels: np.ndarray | None, alpha: float = 0.5) -> np.ndarray:
    rgb = np.repeat(gray_u8[..., None], 3, axis=-1).astype(np.float32)
    if labels is not None:
        colors = np.array(PALETTE, np.float32)[(labels - 1) % len(PALETTE)]
        mask = labels > 0
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * colors[mask]
    return rgb.astype(np.uint8)


def png_bytes(image: np.ndarray, min_width: int = 600) -> bytes:
    """Encode a gray or RGB uint8 array, upscaling small images with nearest neighbour."""
    img = Image.fromarray(image)
    if img.width < min_width:
        factor = int(math.ceil(min_width / img.width))
        img = img.resize((img.width * factor, img.height * factor), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def projection_png(movie: Movie, *, downsample: int = 1, labels: np.ndarray | None = None,
                   n_frames: int = 12) -> bytes:
    """Montage of the max projection, optionally tinted by an ROI label volume."""
    volume = max_projection(movie, n_frames=n_frames, downsample=downsample)
    gray, _ = montage(to_uint8(volume))
    label_tiles = montage(labels.astype(np.int32))[0] if labels is not None else None
    return png_bytes(overlay_rgb(gray, label_tiles))
