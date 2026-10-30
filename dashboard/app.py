"""FastAPI console: node status, dataset browser, result viewer, DAG launcher.

Runs inside the ``dashboard`` container (see docker-compose.yaml) or locally
with ``PYTHONPATH=. uvicorn dashboard.app:app --reload`` and the environment
variables below pointing at a running stack.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from roi_pipeline import datasets, memory, preview

from .airflow_client import AirflowClient, AirflowError
from .nodes import NodeMonitor

BASE = Path(__file__).parent
DAG_IDS = ["roi_factorization", "generate_synthetic_movie"]
RESULT_FILES = ("plan", "factorization", "rois", "classification", "summary")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PREVIEW_BUDGET_BYTES = 256 * 2**20

CONFIG = {
    "airflow_public_url": os.environ.get("AIRFLOW_PUBLIC_URL", "http://localhost:8080").rstrip("/"),
    "flower_public_url": os.environ.get("FLOWER_PUBLIC_URL", "http://localhost:5555").rstrip("/"),
    "data_node_mem_limit": os.environ.get("DATA_NODE_MEM_LIMIT"),
    "poll_seconds": float(os.environ.get("NODE_POLL_SECONDS", "5")),
}

monitor = NodeMonitor(CONFIG["poll_seconds"])
airflow = AirflowClient(os.environ.get("AIRFLOW_API_URL", "http://localhost:8080"),
                        os.environ.get("AIRFLOW_USERNAME", "airflow"),
                        os.environ.get("AIRFLOW_PASSWORD", "airflow"))


class TTLCache:
    """Tiny thread-safe cache for things that are slow to rebuild (previews, DAG params)."""

    def __init__(self, ttl: float, max_items: int = 32):
        self.ttl, self.max_items = ttl, max_items
        self._items: dict = {}
        self._lock = threading.Lock()

    def get(self, key, builder):
        now = time.time()
        with self._lock:
            hit = self._items.get(key)
            if hit and hit[1] > now:
                return hit[0]
        value = builder()
        with self._lock:
            if len(self._items) >= self.max_items:
                self._items.pop(next(iter(self._items)))
            self._items[key] = (value, now + self.ttl)
        return value


preview_cache = TTLCache(ttl=3600)
params_cache = TTLCache(ttl=60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    monitor.start()
    yield


app = FastAPI(title="ROI pipeline console", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
templates.env.filters["size"] = memory.format_size
templates.env.filters["ts"] = lambda value: (
    datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if value else "")


def _safe(fn, default):
    try:
        return fn(), None
    except AirflowError as exc:
        return default, str(exc)


def _check_name(value: str) -> str:
    if not SAFE_NAME.match(value) or ".." in value:
        raise HTTPException(400, f"bad name {value!r}")
    return value


def _dag_params() -> tuple[dict, str | None]:
    def build():
        out, error = {}, None
        for dag_id in DAG_IDS:
            try:
                out[dag_id] = airflow.dag_params(dag_id)
            except AirflowError as exc:
                out[dag_id] = {}
                error = str(exc)
        return out, error

    return params_cache.get("dag_params", build)


def _run_view(run: dict) -> dict:
    conf = run.get("conf") or {}
    dataset = conf.get("dataset") if run["dag_id"] == "roi_factorization" else conf.get("name")
    return {
        **run,
        "dataset": dataset,
        "result_dir": datasets.safe_run_id(run["dag_run_id"]),
        "conf_json": json.dumps(conf, sort_keys=True),
        "airflow_url": f"{CONFIG['airflow_public_url']}/dags/{run['dag_id']}/runs/{run['dag_run_id']}",
    }


def _coerce(form, schema: dict) -> dict:
    """Form fields -> typed DAG conf, using the Param schemas from Airflow."""
    conf = {}
    for name, spec in schema.items():
        kind = (spec.get("schema") or {}).get("type")
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), None)
        raw = form.get(name)
        if kind == "boolean":
            conf[name] = raw in ("on", "true", "1")
            continue
        if raw is None or raw == "":
            continue
        try:
            conf[name] = int(raw) if kind == "integer" else float(raw) if kind == "number" else raw
        except ValueError as exc:
            raise HTTPException(400, f"{name}: {exc}") from exc
    return conf


def _preview_frames(movie: datasets.Movie) -> int:
    frame_bytes = movie.n_voxels * 4
    return int(max(2, min(12, PREVIEW_BUDGET_BYTES // max(frame_bytes, 1))))


def _run_detail(name: str, run: str) -> dict:
    folder = datasets.results_dir() / name / run
    if not folder.is_dir():
        raise HTTPException(404, "no such run")
    detail = {"run": run, "path": str(folder)}
    for key in RESULT_FILES:
        file = folder / f"{key}.json"
        detail[key] = datasets.read_json(file) if file.exists() else None
    rows = []
    if detail["rois"]:
        by_id = {r["id"]: r for r in (detail["classification"] or {}).get("rois", [])}
        for roi in detail["rois"]["rois"]:
            cls = by_id.get(roi["id"], {})
            rows.append({**roi, "quality": cls.get("quality"), "features": cls.get("features", {}),
                         "responsive_stimuli": cls.get("responsive_stimuli"),
                         "response_class": cls.get("response_class")})
    detail["roi_rows"] = rows
    summary = detail["summary"]
    detail["state"] = summary["state"] if summary else "incomplete"
    return detail


# ---------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
def index(request: Request, msg: str | None = None, err: str | None = None):
    health, health_err = _safe(airflow.health, None)
    runs, runs_err = _safe(lambda: airflow.recent_runs(DAG_IDS, 20), [])
    dag_params, params_err = _dag_params()
    return templates.TemplateResponse(request, "index.html", {
        "config": CONFIG,
        "nodes": monitor.snapshot,
        "health": health,
        "airflow_error": health_err or runs_err or params_err,
        "runs": [_run_view(r) for r in runs],
        "datasets": datasets.list_datasets(),
        "dag_params": dag_params,
        "dag_ids": DAG_IDS,
        "msg": msg,
        "err": err,
    })


@app.get("/datasets/{name}", response_class=HTMLResponse)
def dataset_page(request: Request, name: str, run: str | None = None):
    _check_name(name)
    try:
        movie = datasets.open_movie(datasets.resolve_dataset(name))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    results = datasets.list_results(name)
    selected = run or (results[0]["run"] if results else None)
    detail = _run_detail(name, _check_name(selected)) if selected else None
    truth = movie.attrs.get("ground_truth") or {}
    return templates.TemplateResponse(request, "dataset.html", {
        "config": CONFIG,
        "name": name,
        "info": movie.info(),
        "schedule": movie.stimulus_schedule(),
        "n_truth": len(truth.get("rois", [])),
        "results": results,
        "detail": detail,
    })


@app.post("/launch")
async def launch(request: Request):
    form = await request.form()
    dag_id = form.get("dag_id")
    if dag_id not in DAG_IDS:
        raise HTTPException(400, "unknown dag")
    dag_params, _ = _dag_params()
    schema = dag_params.get(dag_id) or {}
    if schema:
        conf = _coerce(form, schema)
    else:
        try:
            conf = json.loads(form.get("conf_json") or "{}")
        except ValueError as exc:
            return RedirectResponse(f"/?err=conf is not valid JSON: {exc}", status_code=303)
    try:
        airflow.unpause(dag_id)
        run = airflow.trigger(dag_id, conf, note="triggered from the dashboard")
    except AirflowError as exc:
        return RedirectResponse(f"/?err={exc}", status_code=303)
    return RedirectResponse(f"/?msg=started {dag_id} run {run['dag_run_id']}", status_code=303)


# ---------------------------------------------------------------- JSON API


@app.get("/api/nodes")
def api_nodes():
    return {**monitor.snapshot, "data_node_mem_limit": CONFIG["data_node_mem_limit"]}


@app.get("/api/health")
def api_health():
    health, error = _safe(airflow.health, None)
    return {"health": health, "error": error}


@app.get("/api/datasets")
def api_datasets():
    return datasets.list_datasets()


@app.get("/api/runs")
def api_runs(limit: int = 20):
    runs, error = _safe(lambda: airflow.recent_runs(DAG_IDS, limit), [])
    return {"runs": [_run_view(r) for r in runs], "error": error}


@app.get("/api/runs/{dag_id}/{run_id}/tasks")
def api_tasks(dag_id: str, run_id: str):
    tasks, error = _safe(lambda: airflow.task_instances(dag_id, run_id), [])
    return {"tasks": tasks, "error": error}


@app.post("/api/launch")
async def api_launch(request: Request):
    body = await request.json()
    dag_id = body.get("dag_id")
    if dag_id not in DAG_IDS:
        raise HTTPException(400, "unknown dag")
    try:
        airflow.unpause(dag_id)
        return airflow.trigger(dag_id, body.get("conf") or {}, note=body.get("note"))
    except AirflowError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/datasets/{name}/preview.png")
def api_preview(name: str, downsample: int = 1):
    _check_name(name)
    try:
        path = datasets.resolve_dataset(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    key = ("preview", str(path), path.stat().st_mtime, downsample)

    def build():
        movie = datasets.open_movie(path)
        return preview.projection_png(movie, downsample=downsample, n_frames=_preview_frames(movie))

    return Response(preview_cache.get(key, build), media_type="image/png")


@app.get("/api/results/{name}/{run}/overlay.png")
def api_overlay(name: str, run: str):
    _check_name(name), _check_name(run)
    folder = datasets.results_dir() / name / run
    if not (folder / "labels.npy").exists():
        raise HTTPException(404, "run has no labels yet")
    key = ("overlay", str(folder), (folder / "labels.npy").stat().st_mtime)

    def build():
        movie = datasets.open_movie(datasets.resolve_dataset(name))
        labels = np.load(folder / "labels.npy")
        downsample = datasets.read_json(folder / "rois.json")["downsample"]
        return preview.projection_png(movie, downsample=downsample, labels=labels, n_frames=_preview_frames(movie))

    return Response(preview_cache.get(key, build), media_type="image/png")


@app.get("/api/results/{name}/{run}/traces.json")
def api_traces(name: str, run: str, ids: str = "", dff: int = 1, max_points: int = 2000):
    _check_name(name), _check_name(run)
    folder = datasets.results_dir() / name / run
    if not (folder / "traces.npy").exists():
        raise HTTPException(404, "run has no traces yet")
    traces = np.load(folder / "traces.npy")
    if dff:
        from roi_pipeline.rois import dff as to_dff

        traces = to_dff(traces)
    wanted = [int(i) for i in ids.split(",") if i.strip().isdigit()] or list(range(1, min(6, traces.shape[1]) + 1))
    wanted = [i for i in wanted if 1 <= i <= traces.shape[1]]
    step = max(1, traces.shape[0] // max_points)
    plan = datasets.read_json(folder / "plan.json") if (folder / "plan.json").exists() else {}
    frame_rate = (plan.get("dataset") or {}).get("frame_rate_hz")
    return {
        "ids": wanted,
        "frame_rate_hz": frame_rate,
        "t": list(range(0, traces.shape[0], step)),
        "traces": {str(i): traces[::step, i - 1].round(4).tolist() for i in wanted},
        "stimuli": plan.get("stimuli"),
        "dff": bool(dff),
    }


@app.get("/api/results/{name}/{run}/{file}.json")
def api_result_file(name: str, run: str, file: str):
    _check_name(name), _check_name(run)
    if file not in RESULT_FILES:
        raise HTTPException(404, "unknown result file")
    path = datasets.results_dir() / name / run / f"{file}.json"
    if not path.exists():
        raise HTTPException(404, "not written yet")
    return JSONResponse(datasets.read_json(path))
