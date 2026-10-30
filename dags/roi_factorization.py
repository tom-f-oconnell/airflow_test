"""Extract ROIs from a movie by matrix factorization, then classify them.

Task placement (Celery queues)::

    plan -> factorize_movie -> extract_rois      queue "data"    (memory-capped data node)
                                  -> classify -> finalize   queue "default" (compute workers)

Only the data node can read the movie. Everything downstream works from the
small artefacts written to ``/data/results/<dataset>/<run>/``; XCom carries
nothing but short dicts.

Memory: ``plan`` compares the data matrix against the node's cgroup cap and
picks in-memory or chunked solvers (``memory_mode=auto``). Forcing
``in_memory`` past the cap is allowed so the OOM kill (task fails, worker
survives) can be observed.
"""
from __future__ import annotations

import logging

import pendulum
from airflow.sdk import Param, dag, task

log = logging.getLogger("airflow.task")


@dag(
    dag_id="roi_factorization",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["roi", "factorization"],
    doc_md=__doc__,
    max_active_runs=1,
    params={
        "dataset": Param("synthetic_small", type="string", description="Name under /data/datasets"),
        "method": Param("nmf", enum=["nmf", "pca_ica"], description="Factorization family"),
        "n_components": Param(20, type="integer", minimum=1, description="Rank k"),
        "memory_mode": Param("auto", enum=["auto", "in_memory", "chunked"],
                             description="auto: in-memory if it fits the cap, else chunked"),
        "spatial_downsample": Param(1, type="integer", minimum=1, description="Block-mean factor in Y and X"),
        "chunk_frames": Param(100, type="integer", minimum=1, description="Frames per read / mini-batch"),
        "threshold_sigma": Param(3.0, type="number", minimum=0.5, description="Spatial map z-score threshold"),
        "min_roi_voxels": Param(20, type="integer", minimum=1),
        "max_rois_per_component": Param(3, type="integer", minimum=1),
        "n_response_classes": Param(3, type="integer", minimum=1, description="k-means groups of responsive ROIs"),
        "z_threshold": Param(2.5, type="number", description="Responsiveness z-score cut-off"),
        "seed": Param(0, type="integer"),
    },
)
def roi_factorization():
    @task(queue="data")
    def plan(**context) -> dict:
        import os
        import socket

        import numpy as np

        from roi_pipeline import datasets, factorize, memory

        params = context["params"]
        run_id = context["run_id"]
        movie = datasets.open_movie(datasets.resolve_dataset(params["dataset"]))
        out = datasets.run_dir(movie.name, run_id, create=True)
        frame_shape = factorize.downsampled_shape(movie.frame_shape, params["spatial_downsample"])
        mem = memory.plan_memory(movie.n_frames, int(np.prod(frame_shape)),
                                 requested_mode=params["memory_mode"], chunk_frames=params["chunk_frames"],
                                 n_components=params["n_components"])
        log.info("%s: %s", movie.name, mem.describe())
        if mem.mode == "in_memory" and not mem.fits_in_memory:
            log.warning("in-memory estimate exceeds the cap; proceeding anyway because memory_mode=in_memory")
        if mem.mode == "chunked" and not mem.fits_chunked:
            raise RuntimeError(f"even chunked processing does not fit: {mem.describe()}; "
                               "lower chunk_frames, raise spatial_downsample or raise DATA_NODE_MEM_LIMIT")
        datasets.write_json(out / "plan.json", {
            "state": "running",
            "dag_id": context["ti"].dag_id,
            "airflow_run_id": run_id,
            "started": pendulum.now("UTC").isoformat(),
            "dataset": movie.info(),
            "params": dict(params),
            "memory": mem.to_dict(),
            "node": {"hostname": socket.gethostname(), "role": os.environ.get("ROI_NODE_ROLE", "unknown")},
            "stimuli": movie.stimulus_schedule(),
            "ground_truth": movie.attrs.get("ground_truth"),
        })
        return {"dataset": movie.name, "run_dir": str(out), "mode": mem.mode, "chunk_frames": mem.chunk_frames}

    @task(queue="data")
    def factorize_movie(plan: dict, **context) -> dict:
        from pathlib import Path

        import zarr

        from roi_pipeline import datasets, factorize, memory

        params = context["params"]
        out = Path(plan["run_dir"])
        movie = datasets.open_movie(datasets.resolve_dataset(plan["dataset"]))
        result = factorize.run_factorization(
            movie, method=params["method"], n_components=params["n_components"], mode=plan["mode"],
            downsample=params["spatial_downsample"], chunk_frames=plan["chunk_frames"], seed=params["seed"])
        group = zarr.open_group(str(out / "components.zarr"), mode="w")
        spatial = group.create_array("spatial", shape=result.spatial.shape, dtype="float32",
                                     chunks=(1, *result.spatial.shape[1:]))
        spatial[:] = result.spatial
        temporal = group.create_array("temporal", shape=result.temporal.shape, dtype="float32")
        temporal[:] = result.temporal
        doc = {"method": result.method, "mode": result.mode, "downsample": result.downsample,
               "n_components": result.n_components, "spatial_shape": list(result.spatial.shape),
               "stats": result.stats}
        datasets.write_json(out / "factorization.json", doc)
        log.info("factorization done in %.1fs, peak RSS %s", result.stats["seconds"],
                 memory.format_size(result.stats["peak_rss_bytes"]))
        return {"n_components": result.n_components, "seconds": result.stats["seconds"],
                "peak_rss_bytes": result.stats["peak_rss_bytes"]}

    @task(queue="data")
    def extract_rois(plan: dict, **context) -> dict:
        from pathlib import Path

        import numpy as np
        import zarr

        from roi_pipeline import datasets, rois

        params = context["params"]
        out = Path(plan["run_dir"])
        movie = datasets.open_movie(datasets.resolve_dataset(plan["dataset"]))
        spatial = np.asarray(zarr.open_group(str(out / "components.zarr"), mode="r")["spatial"][:])
        labels, found = rois.segment_components(
            spatial, threshold_sigma=params["threshold_sigma"], min_voxels=params["min_roi_voxels"],
            max_rois_per_component=params["max_rois_per_component"])
        traces = rois.roi_traces(movie, labels, downsample=params["spatial_downsample"],
                                 chunk_frames=plan["chunk_frames"])
        np.save(out / "labels.npy", labels)
        np.save(out / "traces.npy", traces)
        datasets.write_json(out / "rois.json", {"downsample": params["spatial_downsample"],
                                                "frame_shape": list(labels.shape),
                                                "rois": [r.to_dict() for r in found]})
        log.info("%d ROIs from %d components", len(found), spatial.shape[0])
        return {"n_rois": len(found)}

    @task
    def classify(plan: dict, **context) -> dict:
        from pathlib import Path

        import numpy as np

        from roi_pipeline import classify as classify_mod
        from roi_pipeline import datasets, rois

        params = context["params"]
        out = Path(plan["run_dir"])
        plan_doc = datasets.read_json(out / "plan.json")
        roi_list = datasets.read_json(out / "rois.json")["rois"]
        traces = np.load(out / "traces.npy")
        result = classify_mod.classify_rois(
            roi_list, rois.dff(traces), plan_doc.get("stimuli"),
            n_response_classes=params["n_response_classes"], z_threshold=params["z_threshold"],
            model_path=datasets.models_dir() / "roi_quality.joblib", seed=params["seed"])
        datasets.write_json(out / "classification.json", result)
        return {"quality_counts": result["quality_counts"],
                "n_responsive": result.get("stimuli", {}).get("n_responsive")}

    @task
    def finalize(plan: dict, factorization: dict, extraction: dict, classification: dict, **context) -> dict:
        from pathlib import Path

        import numpy as np

        from roi_pipeline import datasets

        out = Path(plan["run_dir"])
        plan_doc = datasets.read_json(out / "plan.json")
        summary = {
            "state": "success",
            "dataset": plan["dataset"],
            "airflow_run_id": plan_doc["airflow_run_id"],
            "finished": pendulum.now("UTC").isoformat(),
            "params": plan_doc["params"],
            "memory": plan_doc["memory"],
            "node": plan_doc["node"],
            "factorization": factorization,
            "n_rois": extraction["n_rois"],
            "classification": classification,
        }
        truth = plan_doc.get("ground_truth")
        if truth and extraction["n_rois"]:
            factor = plan_doc["params"]["spatial_downsample"]
            found = np.array([r["centroid"] for r in datasets.read_json(out / "rois.json")["rois"]])
            hits = 0
            for roi in truth["rois"]:
                z, y, x = roi["center"]
                dist = np.linalg.norm(found - np.array([z, y / factor, x / factor]), axis=1)
                hits += int(dist.min() <= 3.0)
            summary["ground_truth"] = {"n_planted": len(truth["rois"]), "recovered": hits,
                                       "recall": hits / len(truth["rois"])}
        datasets.write_json(out / "summary.json", summary)
        return {"n_rois": extraction["n_rois"], "recall": summary.get("ground_truth", {}).get("recall")}

    plan_info = plan()
    factorization = factorize_movie(plan_info)
    extraction = extract_rois(plan_info)
    classification = classify(plan_info)
    factorization >> extraction >> classification
    finalize(plan_info, factorization, extraction, classification)


roi_factorization()
