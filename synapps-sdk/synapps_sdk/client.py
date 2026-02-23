"""Synchronous SynApps client (uses httpx)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from synapps_sdk._base import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, _build_headers, _extract_error
from synapps_sdk.exceptions import SynAppsAPIError, SynAppsConnectionError, SynAppsTimeoutError


class SynApps:
    """Synchronous Python client for the SynApps API.

    Usage::

        client = SynApps(base_url="http://localhost:8000/api/v1", api_key="sk-...")
        health = client.get_health()
        result = client.run_template("2brain", input_data={"text": "hello"})
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self._headers = _build_headers(api_key)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SynApps":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise SynAppsConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise SynAppsTimeoutError(str(exc)) from exc

        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"detail": resp.text}
            raise SynAppsAPIError(resp.status_code, _extract_error(resp.status_code, body), body)

        if resp.status_code == 204:
            return None
        return resp.json()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        """GET /health — service health check."""
        return self._request("GET", "/health")

    def get_health_detailed(self) -> Dict[str, Any]:
        """GET /health/detailed — detailed health with DB and provider status."""
        return self._request("GET", "/health/detailed")

    def get_metrics(self) -> Dict[str, Any]:
        """GET /metrics — request and provider metrics."""
        return self._request("GET", "/metrics")

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def list_templates(self) -> List[Dict[str, Any]]:
        """GET /templates — list all imported templates."""
        data = self._request("GET", "/templates")
        return data.get("templates", [])

    def get_template(self, template_id: str, version: Optional[int] = None) -> Dict[str, Any]:
        """GET /templates/{id}/export — export a template definition."""
        params = {}
        if version is not None:
            params["version"] = version
        return self._request("GET", f"/templates/{template_id}/export", params=params)

    def import_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /templates/import — import a template definition."""
        return self._request("POST", "/templates/import", json=template_data)

    def list_template_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """GET /templates/{id}/versions — list all versions of a template."""
        data = self._request("GET", f"/templates/{template_id}/versions")
        return data.get("versions", [])

    def validate_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /templates/validate — validate a template without running it."""
        return self._request("POST", "/templates/validate", json=template_data)

    # ------------------------------------------------------------------
    # Template execution (async task)
    # ------------------------------------------------------------------

    def run_template(
        self,
        template_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST /templates/{id}/run-async — run a template, return task stub."""
        return self._request(
            "POST",
            f"/templates/{template_id}/run-async",
            json={"input": input_data or {}},
        )

    def run_template_and_poll(
        self,
        template_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Run a template and poll until completion or timeout."""
        task = self.run_template(template_id, input_data)
        task_id = task["task_id"]
        return self.poll_task(task_id, poll_interval=poll_interval, timeout=timeout)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """GET /tasks/{id} — get task status and result."""
        return self._request("GET", f"/tasks/{task_id}")

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """GET /tasks — list tasks, optionally filtered by status."""
        params = {}
        if status:
            params["status"] = status
        data = self._request("GET", "/tasks", params=params)
        return data.get("tasks", data) if isinstance(data, dict) else data

    def poll_task(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Poll a task until it reaches a terminal state (completed/failed)."""
        deadline = time.monotonic() + timeout
        while True:
            task = self.get_task(task_id)
            if task.get("status") in ("completed", "failed"):
                return task
            if time.monotonic() >= deadline:
                raise SynAppsTimeoutError(
                    f"Task {task_id} did not complete within {timeout}s"
                )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Flows
    # ------------------------------------------------------------------

    def create_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """POST /flows — create or update a flow."""
        return self._request("POST", "/flows", json=flow)

    def list_flows(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """GET /flows — list flows with pagination."""
        return self._request("GET", "/flows", params={"page": page, "page_size": page_size})

    def get_flow(self, flow_id: str) -> Dict[str, Any]:
        """GET /flows/{id} — get a flow by ID."""
        return self._request("GET", f"/flows/{flow_id}")

    def delete_flow(self, flow_id: str) -> None:
        """DELETE /flows/{id} — delete a flow."""
        self._request("DELETE", f"/flows/{flow_id}")

    def export_flow(self, flow_id: str) -> Dict[str, Any]:
        """GET /flows/{id}/export — export a flow definition."""
        return self._request("GET", f"/flows/{flow_id}/export")

    def import_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /flows/import — import a flow definition."""
        return self._request("POST", "/flows/import", json=flow_data)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def run_flow(self, flow_id: str) -> Dict[str, Any]:
        """POST /flows/{id}/runs — execute a flow."""
        return self._request("POST", f"/flows/{flow_id}/runs")

    def list_runs(self) -> List[Dict[str, Any]]:
        """GET /runs — list workflow runs."""
        data = self._request("GET", "/runs")
        return data.get("runs", data) if isinstance(data, dict) else data

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """GET /runs/{id} — get a run by ID."""
        return self._request("GET", f"/runs/{run_id}")

    def get_run_trace(self, run_id: str) -> Dict[str, Any]:
        """GET /runs/{id}/trace — get execution trace for a run."""
        return self._request("GET", f"/runs/{run_id}/trace")

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def list_providers(self) -> List[Dict[str, Any]]:
        """GET /providers — list all discovered LLM providers."""
        data = self._request("GET", "/providers")
        return data.get("providers", data) if isinstance(data, dict) else data

    def get_provider_health(self, name: str) -> Dict[str, Any]:
        """GET /providers/{name}/health — check a provider's health."""
        return self._request("GET", f"/providers/{name}/health")

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(
        self,
        status: Optional[str] = None,
        template_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """GET /history — workflow execution history."""
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if template_id:
            params["template_id"] = template_id
        return self._request("GET", "/history", params=params)
