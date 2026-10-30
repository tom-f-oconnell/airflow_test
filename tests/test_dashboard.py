"""Smoke test of the console against artefacts shaped like the DAG's output."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Nothing listens on the discard port; the console must cope with Airflow being absent.
os.environ.setdefault("AIRFLOW_API_URL", "http://127.0.0.1:9")

from roi_pipeline import classify, datasets, factorize, memory, rois

RUN_ID = "manual__2026-10-29T10:00:00+00:00"


@pytest.fixture(scope="module")
def run_name(data_root, small_movie_path) -> str:
    movie = datasets.open_movie(small_movie_path)
    out = datasets.run_dir(movie.name, RUN_ID, create=True)
    plan = memory.plan_memory(movie.n_frames, movie.n_voxels, chunk_frames=40)
    datasets.write_json(out / "plan.json", {
        "state": "running", "dag_id": "roi_factorization", "airflow_run_id": RUN_ID,
        "dataset": movie.info(), "params": {"method": "nmf", "spatial_downsample": 1},
        "memory": plan.to_dict(), "node": {"hostname": "data-node", "role": "data"},
        "stimuli": movie.stimulus_schedule(), "ground_truth": movie.attrs["ground_truth"]})
    result = factorize.run_factorization(movie, method="nmf", n_components=8, mode=plan.mode, chunk_frames=40)
    datasets.write_json(out / "factorization.json", {"method": "nmf", "mode": plan.mode, "downsample": 1,
                                                      "n_components": 8, "stats": result.stats})
    labels, found = rois.segment_components(result.spatial, threshold_sigma=3.0, min_voxels=8)
    traces = rois.roi_traces(movie, labels, chunk_frames=40)
    np.save(out / "labels.npy", labels)
    np.save(out / "traces.npy", traces)
    datasets.write_json(out / "rois.json", {"downsample": 1, "frame_shape": list(labels.shape),
                                            "rois": [r.to_dict() for r in found]})
    cls = classify.classify_rois([r.to_dict() for r in found], rois.dff(traces), movie.stimulus_schedule())
    datasets.write_json(out / "classification.json", cls)
    datasets.write_json(out / "summary.json", {"state": "success", "n_rois": len(found),
                                               "ground_truth": {"n_planted": 5, "recovered": 4, "recall": 0.8}})
    return out.name


def test_console_pages_and_api(run_name):
    from dashboard.app import app

    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "synthetic_test" in page.text and "Airflow API" in page.text  # Airflow is not running here

        page = client.get("/datasets/synthetic_test")
        assert page.status_code == 200
        assert run_name in page.text and "ROI traces" in page.text and "recall" in page.text

        png = client.get("/api/datasets/synthetic_test/preview.png")
        assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
        overlay = client.get(f"/api/results/synthetic_test/{run_name}/overlay.png")
        assert overlay.status_code == 200 and overlay.content[:4] == b"\x89PNG"

        traces = client.get(f"/api/results/synthetic_test/{run_name}/traces.json", params={"ids": "1,2"}).json()
        assert traces["ids"] == [1, 2] and len(traces["t"]) == 160 and traces["stimuli"]["ids"] == ["0", "1"]
        summary = client.get(f"/api/results/synthetic_test/{run_name}/summary.json").json()
        assert summary["state"] == "success"
        assert client.get(f"/api/results/synthetic_test/{run_name}/secrets.json").status_code == 404

        assert "nodes" in client.get("/api/nodes").json()
        assert [d["name"] for d in client.get("/api/datasets").json()] == ["synthetic_test"]
        assert client.get("/api/health").json()["error"]
        assert client.get("/datasets/nope").status_code == 404
        assert client.get("/datasets/..%2Fetc").status_code in (400, 404)

        launch = client.post("/launch", data={"dag_id": "roi_factorization", "dataset": "synthetic_test"},
                             follow_redirects=False)
        assert launch.status_code == 303 and "err=" in launch.headers["location"]
        assert client.post("/api/launch", json={"dag_id": "roi_factorization", "conf": {}}).status_code == 502
