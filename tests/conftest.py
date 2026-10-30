import os
from pathlib import Path

import pytest

from roi_pipeline import synthetic

SMALL = dict(n_frames=160, n_z=4, height=32, width=32, n_rois=5, n_stimuli=2, n_trials=3,
             stim_frames=6, pre_frames=10, post_frames=20, noise_sd=20.0, seed=1, chunk_frames=16)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    """A throwaway ROI_DATA_ROOT with datasets/, results/ and models/."""
    root = tmp_path_factory.mktemp("data")
    for sub in ("datasets", "results", "models"):
        (root / sub).mkdir()
    os.environ["ROI_DATA_ROOT"] = str(root)
    return root


@pytest.fixture(scope="session")
def small_movie_path(data_root) -> Path:
    path = data_root / "datasets" / "synthetic_test.zarr"
    synthetic.generate_synthetic_movie(path, **SMALL)
    return path
