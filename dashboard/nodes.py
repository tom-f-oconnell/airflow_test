"""Container status via the Docker socket, refreshed by a background thread.

Each node row carries the configured memory cap (HostConfig.Memory), the live
usage reported by the cgroup (minus reclaimable file cache, as `docker stats`
does), CPU percentage and the Compose health state.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

ROLE_BY_SERVICE = {
    "data-node": "data node",
    "compute-worker": "compute worker",
    "airflow-apiserver": "airflow",
    "airflow-scheduler": "airflow",
    "airflow-dag-processor": "airflow",
    "airflow-triggerer": "airflow",
    "airflow-init": "airflow",
    "flower": "airflow",
    "postgres": "storage",
    "redis": "storage",
    "dashboard": "dashboard",
}
SERVICE_ORDER = ["data-node", "compute-worker", "airflow-apiserver", "airflow-scheduler",
                 "airflow-dag-processor", "airflow-triggerer", "flower", "postgres", "redis", "dashboard",
                 "airflow-init", "airflow-cli"]


def cpu_percent(stats: dict) -> float | None:
    try:
        cpu, pre = stats["cpu_stats"], stats["precpu_stats"]
        cpu_delta = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
        sys_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
        n_cpu = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or []) or 1
    except (KeyError, TypeError):
        return None
    if sys_delta <= 0 or cpu_delta < 0:
        return 0.0
    return round(cpu_delta / sys_delta * n_cpu * 100, 1)


def memory_usage(stats: dict) -> tuple[int | None, int | None]:
    mem = stats.get("memory_stats") or {}
    usage, limit = mem.get("usage"), mem.get("limit")
    if usage is None:
        return None, limit
    detail = mem.get("stats") or {}
    cache = detail.get("inactive_file", detail.get("total_inactive_file", 0))
    return max(usage - cache, 0), limit


class NodeMonitor:
    """Polls Docker every ``poll_seconds`` and keeps the latest snapshot."""

    def __init__(self, poll_seconds: float = 5.0):
        self.poll_seconds = poll_seconds
        self.snapshot = {"nodes": [], "error": None, "updated": None, "project": None}
        self._client = None
        self._project = None
        self._thread = threading.Thread(target=self._loop, name="node-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _connect(self):
        import docker

        if self._client is None:
            self._client = docker.from_env()
            self._project = self._find_project()
        return self._client

    def _find_project(self) -> str | None:
        """Compose project of the container we run in (falls back to COMPOSE_PROJECT_NAME)."""
        try:
            me = self._client.containers.get(socket.gethostname())
            return me.labels.get("com.docker.compose.project")
        except Exception:  # noqa: BLE001 - not running in a container
            return os.environ.get("COMPOSE_PROJECT_NAME")

    def _loop(self) -> None:
        while True:
            started = time.time()
            try:
                self.snapshot = self._refresh()
            except Exception as exc:  # noqa: BLE001 - keep polling
                log.warning("node monitor: %s", exc)
                self.snapshot = {"nodes": [], "error": f"{type(exc).__name__}: {exc}",
                                 "updated": time.time(), "project": self._project}
                self._client = None
            time.sleep(max(0.5, self.poll_seconds - (time.time() - started)))

    def _refresh(self) -> dict:
        client = self._connect()
        filters = {"label": f"com.docker.compose.project={self._project}"} if self._project else None
        containers = client.containers.list(all=True, filters=filters)
        with ThreadPoolExecutor(max_workers=8) as pool:
            nodes = list(pool.map(self._describe, containers))
        nodes.sort(key=lambda n: (SERVICE_ORDER.index(n["service"]) if n["service"] in SERVICE_ORDER else 99,
                                  n["name"]))
        return {"nodes": nodes, "error": None, "updated": time.time(), "project": self._project}

    def _describe(self, container) -> dict:
        attrs = container.attrs
        labels = container.labels or {}
        service = labels.get("com.docker.compose.service", container.name)
        host_config = attrs.get("HostConfig") or {}
        node = {
            "name": container.name,
            "service": service,
            "replica": labels.get("com.docker.compose.container-number"),
            "role": ROLE_BY_SERVICE.get(service, "other"),
            "status": container.status,
            "health": ((attrs.get("State") or {}).get("Health") or {}).get("Status"),
            "started_at": (attrs.get("State") or {}).get("StartedAt"),
            "restart_count": attrs.get("RestartCount", 0),
            "oom_killed": (attrs.get("State") or {}).get("OOMKilled", False),
            "image": (container.image.tags or [container.image.short_id])[0] if container.image else None,
            "mem_cap_bytes": host_config.get("Memory") or None,
            "cpus": (host_config.get("NanoCpus") or 0) / 1e9 or None,
            "mem_used_bytes": None,
            "mem_limit_bytes": None,
            "cpu_percent": None,
        }
        if container.status == "running":
            try:
                stats = container.stats(stream=False)
                node["mem_used_bytes"], node["mem_limit_bytes"] = memory_usage(stats)
                node["cpu_percent"] = cpu_percent(stats)
            except Exception as exc:  # noqa: BLE001 - container may have just exited
                node["stats_error"] = str(exc)
        return node
