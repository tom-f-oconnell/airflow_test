"""Generate a synthetic volumetric calcium-imaging movie on the data node.

Creates ``/data/datasets/<name>.zarr`` with planted ROIs, a stimulus schedule
and the ground truth in the zarr attrs. Sizes are free, so a movie that does
not fit the data node's memory cap can be made: the generator streams.
"""
from __future__ import annotations

import logging

import pendulum
from airflow.sdk import Param, dag, task

log = logging.getLogger("airflow.task")


@dag(
    dag_id="generate_synthetic_movie",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["roi", "synthetic"],
    doc_md=__doc__,
    params={
        "name": Param("synthetic_small", type="string", description="Dataset name -> /data/datasets/<name>.zarr"),
        "n_frames": Param(300, type="integer", minimum=20, description="Movie length (T)"),
        "n_z": Param(8, type="integer", minimum=1, description="Planes per volume (Z)"),
        "height": Param(64, type="integer", minimum=16, description="Rows per plane (Y)"),
        "width": Param(64, type="integer", minimum=16, description="Columns per plane (X)"),
        "n_rois": Param(12, type="integer", minimum=1, description="Planted Gaussian blobs"),
        "n_stimuli": Param(3, type="integer", minimum=0, description="Distinct stimuli (0 = none)"),
        "n_trials": Param(4, type="integer", minimum=1, description="Presentations per stimulus"),
        "stim_frames": Param(8, type="integer", minimum=1, description="Stimulus duration in frames"),
        "noise_sd": Param(25.0, type="number", minimum=0, description="Gaussian noise (uint16 counts)"),
        "seed": Param(0, type="integer"),
        "overwrite": Param(False, type="boolean", description="Replace an existing dataset of the same name"),
    },
)
def generate_synthetic_movie():
    @task(queue="data")
    def generate(**context) -> dict:
        import shutil

        from roi_pipeline import datasets, memory, synthetic

        params = dict(context["params"])
        name = params.pop("name")
        overwrite = params.pop("overwrite")
        path = datasets.datasets_dir() / f"{name}.zarr"
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} exists; set overwrite=true to replace it")
            shutil.rmtree(path)
        n_bytes = params["n_frames"] * params["n_z"] * params["height"] * params["width"] * 2
        log.info("writing %s (%s as uint16) on a node capped at %s", path, memory.format_size(n_bytes),
                 memory.format_size(memory.memory_cap_bytes()))
        attrs = synthetic.generate_synthetic_movie(path, **params)
        return {"dataset": name, "path": str(path), "size_bytes": n_bytes,
                "n_rois": len(attrs["ground_truth"]["rois"]), "stimuli": attrs["stimuli"]["ids"]}

    generate()


generate_synthetic_movie()
