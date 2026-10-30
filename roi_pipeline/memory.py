"""Memory accounting for the data node.

The data node is started with a cgroup memory cap (``DATA_NODE_MEM_LIMIT`` in
docker-compose). These helpers read that cap from inside the container,
estimate what a factorization will need, and pick between loading the whole
movie into memory or streaming it in time chunks.
"""
from __future__ import annotations

import os
import re
import resource
from dataclasses import asdict, dataclass

_CGROUP_V2_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

# Interpreter, Celery worker and imported libraries: resident before any data
# is loaded.
BASE_OVERHEAD_BYTES = 512 * 2**20
# Peak working set of the in-memory solvers relative to the data matrix. NMF
# and randomized SVD both keep a couple of matrix-sized temporaries around.
IN_MEMORY_FACTOR = 3.0
# Fraction of the cap we allow ourselves to plan for.
SAFETY_FRACTION = 0.9

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)
_UNITS = {"": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}


def parse_size(text: str | int | float) -> int:
    """Parse a Docker-style size ("512m", "8g", "1.5G", "1073741824") to bytes."""
    if isinstance(text, (int, float)):
        return int(text)
    match = _SIZE_RE.match(str(text))
    if not match:
        raise ValueError(f"unrecognised size: {text!r}")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def format_size(n: int | float | None) -> str:
    """Human readable binary size ("1.50 GiB"); None renders as "unlimited"."""
    if n is None:
        return "unlimited"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def host_memory_bytes() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def cgroup_memory_limit() -> int | None:
    """Memory cap of the cgroup this process runs in, or None if unlimited."""
    for path in (_CGROUP_V2_MAX, _CGROUP_V1_LIMIT):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        value = int(raw)
        # cgroup v1 reports an enormous number when no limit is set.
        if value >= 2**60:
            return None
        return value
    return None


def memory_cap_bytes() -> int:
    """Effective cap: the cgroup limit if there is one, else physical memory."""
    limit = cgroup_memory_limit()
    host = host_memory_bytes()
    return min(limit, host) if limit else host


def current_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_rss_bytes() -> int:
    """High-water mark of this process' resident set (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


@dataclass
class MemoryPlan:
    """Outcome of :func:`plan_memory`. ``to_dict`` makes it XCom/JSON friendly."""

    cap_bytes: int
    cgroup_limited: bool
    n_frames: int
    n_voxels: int
    matrix_bytes: int
    in_memory_estimate: int
    chunked_estimate: int
    fits_in_memory: bool
    fits_chunked: bool
    requested_mode: str
    mode: str
    chunk_frames: int

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"matrix {format_size(self.matrix_bytes)}; in-memory needs ~{format_size(self.in_memory_estimate)}, "
            f"chunked needs ~{format_size(self.chunked_estimate)}; cap {format_size(self.cap_bytes)} "
            f"({'cgroup' if self.cgroup_limited else 'host'}); mode={self.mode}"
        )


def plan_memory(
    n_frames: int,
    n_voxels: int,
    *,
    requested_mode: str = "auto",
    chunk_frames: int = 100,
    n_components: int = 20,
    itemsize: int = 4,
    cap_bytes: int | None = None,
) -> MemoryPlan:
    """Decide how to run a factorization under the current memory cap.

    Args:
        n_frames: Movie length (rows of the data matrix).
        n_voxels: Voxels per frame after spatial downsampling (columns).
        requested_mode: ``auto`` picks in-memory when it fits and chunked
            otherwise. ``in_memory`` and ``chunked`` force a strategy; forcing
            ``in_memory`` past the cap is allowed on purpose so the OOM
            behaviour of the node can be exercised.
        chunk_frames: Frames per chunk in chunked mode (clamped to n_frames).
        n_components: Factorization rank, which sizes the component matrices.
        itemsize: Bytes per matrix element (float32).
        cap_bytes: Override the cap (tests); default reads the cgroup.
    """
    if cap_bytes is None:
        cap = memory_cap_bytes()
        limited = cgroup_memory_limit() is not None
    else:
        cap, limited = int(cap_bytes), True
    matrix = int(n_frames) * int(n_voxels) * itemsize
    in_memory_estimate = BASE_OVERHEAD_BYTES + int(IN_MEMORY_FACTOR * matrix)
    chunk_frames = max(1, min(int(chunk_frames), int(n_frames)))
    chunk_bytes = chunk_frames * int(n_voxels) * itemsize
    # The chunk, sklearn's working copy of it, and the k x P components held
    # twice over (model + float64 scratch).
    chunked_estimate = BASE_OVERHEAD_BYTES + 3 * chunk_bytes + 2 * n_components * int(n_voxels) * 8
    budget = SAFETY_FRACTION * cap
    fits_in_memory = in_memory_estimate <= budget
    fits_chunked = chunked_estimate <= budget
    if requested_mode == "auto":
        mode = "in_memory" if fits_in_memory else "chunked"
    elif requested_mode in ("in_memory", "chunked"):
        mode = requested_mode
    else:
        raise ValueError(f"unknown memory mode {requested_mode!r}")
    return MemoryPlan(
        cap_bytes=cap,
        cgroup_limited=limited,
        n_frames=int(n_frames),
        n_voxels=int(n_voxels),
        matrix_bytes=matrix,
        in_memory_estimate=in_memory_estimate,
        chunked_estimate=chunked_estimate,
        fits_in_memory=fits_in_memory,
        fits_chunked=fits_chunked,
        requested_mode=requested_mode,
        mode=mode,
        chunk_frames=chunk_frames,
    )
