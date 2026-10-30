"""Locating and reading movie datasets on the data volume.

Layout under ``ROI_DATA_ROOT`` (``/data`` inside the containers)::

    datasets/<name>.zarr       zarr group with array "movie" (T, Z, Y, X) + attrs
    datasets/<name>.npy        plain array, (T, Z, Y, X) or (T, Y, X)
    datasets/<name>.tif        pages in T-major order (T*Z pages) or (T, Y, X)
    datasets/<name>.json       optional sidecar attrs for .npy/.tif (stimuli ...)
    results/<name>/<run>/      outputs of one roi_factorization run
    models/                    optional trained ROI classifiers (joblib)

Movies are always handed out as float32 blocks of frames so the rest of the
pipeline is format agnostic.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

SUPPORTED_SUFFIXES = (".zarr", ".npy", ".tif", ".tiff")


def data_root() -> Path:
    return Path(os.environ.get("ROI_DATA_ROOT", "/data"))


def datasets_dir() -> Path:
    return data_root() / "datasets"


def results_dir() -> Path:
    return data_root() / "results"


def models_dir() -> Path:
    return data_root() / "models"


class Movie:
    """Lazy reader over a (T, Z, Y, X) movie.

    Attributes:
        path: Location on disk.
        shape: (T, Z, Y, X); 3-D sources are reported with Z=1.
        dtype: Storage dtype. Reads are converted to float32.
        attrs: JSON-able metadata (stimulus schedule, ground truth, ...).
    """

    def __init__(self, path: Path, shape: tuple[int, ...], dtype, attrs: dict,
                 reader: Callable[[int, int], np.ndarray]):
        if len(shape) == 3:
            shape = (shape[0], 1, shape[1], shape[2])
        if len(shape) != 4:
            raise ValueError(f"{path}: expected a (T, Z, Y, X) or (T, Y, X) movie, got shape {shape}")
        self.path = Path(path)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.attrs = attrs or {}
        self._reader = reader

    @property
    def name(self) -> str:
        return dataset_name(self.path)

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.shape[1:]

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.frame_shape))

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def read(self, t0: int, t1: int) -> np.ndarray:
        """Frames ``t0:t1`` as float32 (t1 - t0, Z, Y, X)."""
        t0, t1 = max(0, t0), min(self.n_frames, t1)
        block = np.asarray(self._reader(t0, t1))
        if block.ndim == 3:
            block = block[:, None]
        return block.astype(np.float32, copy=False)

    def iter_chunks(self, chunk_frames: int) -> Iterator[tuple[int, np.ndarray]]:
        chunk_frames = max(1, int(chunk_frames))
        for t0 in range(0, self.n_frames, chunk_frames):
            yield t0, self.read(t0, t0 + chunk_frames)

    def sample_frames(self, n: int) -> np.ndarray:
        """``n`` evenly spaced frames as (n, Z, Y, X); cheap preview input."""
        n = max(1, min(int(n), self.n_frames))
        idx = np.linspace(0, self.n_frames - 1, n).round().astype(int)
        return np.stack([self.read(i, i + 1)[0] for i in np.unique(idx)])

    def stimulus_schedule(self) -> dict | None:
        schedule = self.attrs.get("stimuli")
        return schedule if schedule and schedule.get("onsets") else None

    def info(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "format": self.path.suffix.lstrip("."),
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "size_bytes": self.nbytes,
            "n_frames": self.n_frames,
            "n_voxels": self.n_voxels,
            "has_stimuli": self.stimulus_schedule() is not None,
            "has_ground_truth": "ground_truth" in self.attrs,
            "frame_rate_hz": self.attrs.get("frame_rate_hz"),
        }


def dataset_name(path: Path | str) -> str:
    path = Path(path)
    return path.name[: -len(path.suffix)] if path.suffix in SUPPORTED_SUFFIXES else path.name


def _sidecar_attrs(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    return {}


def _open_zarr(path: Path) -> Movie:
    import zarr

    node = zarr.open(str(path), mode="r")
    if isinstance(node, zarr.Group):
        if "movie" not in node:
            raise ValueError(f"{path}: zarr group has no 'movie' array")
        arr = node["movie"]
        attrs = dict(node.attrs)
    else:
        arr = node
        attrs = dict(arr.attrs)
    return Movie(path, arr.shape, arr.dtype, attrs, lambda t0, t1: arr[t0:t1])


def _open_npy(path: Path) -> Movie:
    arr = np.load(path, mmap_mode="r")
    return Movie(path, arr.shape, arr.dtype, _sidecar_attrs(path), lambda t0, t1: arr[t0:t1])


def _open_tiff(path: Path) -> Movie:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        shape, dtype = tuple(series.shape), series.dtype
        n_pages, plane_shape = len(tif.pages), tuple(tif.pages[0].shape)
    if len(shape) not in (3, 4) or len(plane_shape) != 2:
        raise ValueError(f"{path}: expected a 3-D or 4-D single-channel TIFF series, got {shape}")
    n_z = shape[1] if len(shape) == 4 else 1
    if n_pages != shape[0] * n_z:
        raise ValueError(f"{path}: {n_pages} pages do not match T*Z = {shape[0]}*{n_z}; "
                         "pages must be one plane each, T-major then Z")

    def reader(t0: int, t1: int) -> np.ndarray:
        pages = tifffile.imread(str(path), key=range(t0 * n_z, t1 * n_z))
        return np.asarray(pages).reshape(t1 - t0, n_z, *plane_shape)

    return Movie(path, shape, dtype, _sidecar_attrs(path), reader)


def open_movie(path: Path | str) -> Movie:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".zarr":
        return _open_zarr(path)
    if suffix == ".npy":
        return _open_npy(path)
    if suffix in (".tif", ".tiff"):
        return _open_tiff(path)
    raise ValueError(f"{path}: unsupported dataset format (expected one of {SUPPORTED_SUFFIXES})")


def resolve_dataset(name: str, root: Path | None = None) -> Path:
    """Find ``datasets/<name>.<suffix>``; ``name`` may already carry the suffix."""
    root = root or datasets_dir()
    candidate = root / name
    if candidate.suffix in SUPPORTED_SUFFIXES and candidate.exists():
        return candidate
    for suffix in SUPPORTED_SUFFIXES:
        candidate = root / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no dataset named {name!r} under {root}")


def list_datasets(root: Path | None = None) -> list[dict]:
    """Metadata for every dataset under ``datasets/`` (unreadable ones carry an ``error``)."""
    root = root or datasets_dir()
    entries = []
    if not root.exists():
        return entries
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.name.startswith("."):
            continue
        try:
            info = open_movie(path).info()
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            info = {"name": dataset_name(path), "path": str(path), "format": path.suffix.lstrip("."),
                    "error": f"{type(exc).__name__}: {exc}"}
        info["modified"] = path.stat().st_mtime
        info["n_results"] = len(list_results(info["name"]))
        entries.append(info)
    return entries


def safe_run_id(run_id: str) -> str:
    """Airflow run ids contain ':' and '+'; make them usable as directory names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)


def run_dir(dataset: str, run_id: str, create: bool = False) -> Path:
    path = results_dir() / dataset / safe_run_id(run_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def list_results(dataset: str) -> list[dict]:
    """Runs recorded under ``results/<dataset>/``, newest first."""
    base = results_dir() / dataset
    runs = []
    if not base.exists():
        return runs
    for path in base.iterdir():
        if not path.is_dir():
            continue
        entry = {"run": path.name, "path": str(path), "modified": path.stat().st_mtime,
                 "plan": None, "summary": None, "has_classification": (path / "classification.json").exists()}
        for key in ("plan", "summary"):
            file = path / f"{key}.json"
            if file.exists():
                try:
                    entry[key] = read_json(file)
                except (OSError, ValueError):
                    entry[key] = {"error": "unreadable"}
        runs.append(entry)
    runs.sort(key=lambda r: r["modified"], reverse=True)
    return runs
