"""Thin client for the Airflow 3 REST API (v2) with a cached JWT."""
from __future__ import annotations

import threading
import time

import httpx


class AirflowError(RuntimeError):
    pass


def _json(resp: httpx.Response, path: str):
    try:
        return resp.json()
    except ValueError as exc:
        raise AirflowError(f"{path}: not an Airflow response (is something else on this port?): "
                           f"{resp.text[:120]!r}") from exc


class AirflowClient:
    """Minimal wrapper: every method raises :class:`AirflowError` on failure."""

    def __init__(self, base_url: str, username: str, password: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self._token: str | None = None
        self._token_at = 0.0
        self._lock = threading.Lock()

    def _login(self) -> str:
        try:
            resp = httpx.post(f"{self.base_url}/auth/token", timeout=self.timeout,
                              json={"username": self.username, "password": self.password})
        except httpx.HTTPError as exc:
            raise AirflowError(f"cannot reach Airflow at {self.base_url}: {exc}") from exc
        if resp.status_code not in (200, 201):
            raise AirflowError(f"login failed ({resp.status_code}): {resp.text[:200]}")
        return _json(resp, "/auth/token")["access_token"]

    def _headers(self, force: bool = False) -> dict:
        with self._lock:
            # Tokens default to a 24 h lifetime; refresh well before that.
            if force or self._token is None or time.time() - self._token_at > 6 * 3600:
                self._token = self._login()
                self._token_at = time.time()
            return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        for attempt in (0, 1):
            try:
                resp = httpx.request(method, url, headers=self._headers(force=attempt == 1),
                                     timeout=self.timeout, **kwargs)
            except httpx.HTTPError as exc:
                raise AirflowError(f"{method} {path}: {exc}") from exc
            if resp.status_code == 401 and attempt == 0:
                continue
            if resp.status_code >= 400:
                raise AirflowError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return _json(resp, path) if resp.content else None
        raise AirflowError(f"{method} {path}: unauthorised")

    def health(self) -> dict:
        try:
            resp = httpx.get(f"{self.base_url}/api/v2/monitor/health", timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise AirflowError(f"health check failed: {exc}") from exc
        return _json(resp, "/api/v2/monitor/health")

    def dags(self) -> list[dict]:
        return self._request("GET", "/api/v2/dags", params={"limit": 100})["dags"]

    def dag_details(self, dag_id: str) -> dict:
        return self._request("GET", f"/api/v2/dags/{dag_id}/details")

    def dag_params(self, dag_id: str) -> dict:
        """``{name: {"default": ..., "schema": {...}, "description": ...}}`` from the DAG definition."""
        raw = self.dag_details(dag_id).get("params") or {}
        out = {}
        for name, value in raw.items():
            if isinstance(value, dict) and ("schema" in value or "default" in value or "value" in value):
                default = value.get("value", value.get("default"))
                schema = value.get("schema") or {}
                out[name] = {"default": default, "schema": schema,
                             "description": value.get("description") or schema.get("description", "")}
            else:
                out[name] = {"default": value, "schema": {}, "description": ""}
        return out

    def dag_runs(self, dag_id: str, limit: int = 20) -> list[dict]:
        params = {"limit": limit, "order_by": "-run_after"}
        try:
            runs = self._request("GET", f"/api/v2/dags/{dag_id}/dagRuns", params=params)["dag_runs"]
        except AirflowError:
            runs = self._request("GET", f"/api/v2/dags/{dag_id}/dagRuns", params={"limit": limit})["dag_runs"]
        return sorted(runs, key=lambda r: r.get("run_after") or r.get("start_date") or "", reverse=True)

    def recent_runs(self, dag_ids: list[str], limit: int = 20) -> list[dict]:
        runs = []
        for dag_id in dag_ids:
            try:
                runs.extend(self.dag_runs(dag_id, limit))
            except AirflowError as exc:
                # A DAG that has not been parsed yet simply has no runs.
                if "404" not in str(exc):
                    raise
        runs.sort(key=lambda r: r.get("run_after") or r.get("start_date") or "", reverse=True)
        return runs[:limit]

    def task_instances(self, dag_id: str, run_id: str) -> list[dict]:
        data = self._request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
        return sorted(data["task_instances"], key=lambda t: (t.get("start_date") or "9", t["task_id"]))

    def unpause(self, dag_id: str) -> None:
        self._request("PATCH", f"/api/v2/dags/{dag_id}", json={"is_paused": False})

    def trigger(self, dag_id: str, conf: dict, note: str | None = None) -> dict:
        body = {"logical_date": None, "conf": conf}
        if note:
            body["note"] = note
        return self._request("POST", f"/api/v2/dags/{dag_id}/dagRuns", json=body)
