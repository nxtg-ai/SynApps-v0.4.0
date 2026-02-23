"""
Tests for the SynApps SDK client library (sync + async).

Uses the real FastAPI app via httpx transports so tests validate actual API
round-trips without mocking. The TestClient triggers lifespan (DB init).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Make the SDK importable
sdk_path = str(Path(__file__).resolve().parents[3] / "synapps-sdk")
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from synapps_sdk import AsyncSynApps, SynApps  # noqa: E402
from synapps_sdk.exceptions import (  # noqa: E402
    SynAppsAPIError,
    SynAppsConnectionError,
    SynAppsTimeoutError,
)

from apps.orchestrator.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build SDK clients backed by the test app
# ---------------------------------------------------------------------------

def _make_sync_client(test_client: TestClient) -> SynApps:
    """Create a sync SDK client that shares the TestClient's transport."""
    client = SynApps.__new__(SynApps)
    client.base_url = "http://testserver/api/v1"
    client._headers = {"Content-Type": "application/json"}
    # TestClient._transport is an httpx transport wired to ASGI
    client._client = httpx.Client(
        transport=test_client._transport,
        base_url="http://testserver/api/v1",
        headers=client._headers,
    )
    return client


async def _make_async_client() -> AsyncSynApps:
    """Create an async SDK client backed by the ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    client = AsyncSynApps.__new__(AsyncSynApps)
    client.base_url = "http://testserver/api/v1"
    client._headers = {"Content-Type": "application/json"}
    client._client = httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver/api/v1",
        headers=client._headers,
    )
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_app():
    """TestClient triggers lifespan (DB creation)."""
    with TestClient(app) as tc:
        yield tc


@pytest.fixture()
def sync_client(test_app):
    """Sync SDK client using the TestClient transport."""
    client = _make_sync_client(test_app)
    yield client
    client.close()


@pytest_asyncio.fixture()
async def async_client(test_app):
    """Async SDK client backed by ASGI transport."""
    client = await _make_async_client()
    yield client
    await client.close()


MINIMAL_FLOW = {
    "name": "SDK Test Flow",
    "nodes": [
        {"id": "start-1", "type": "StartNode", "data": {"label": "Start"}, "position": {"x": 0, "y": 0}},
        {"id": "end-1", "type": "EndNode", "data": {"label": "End"}, "position": {"x": 200, "y": 0}},
    ],
    "edges": [{"id": "e1", "source": "start-1", "target": "end-1"}],
}


# ---------------------------------------------------------------------------
# Sync client tests
# ---------------------------------------------------------------------------

class TestSyncClient:

    def test_get_health(self, sync_client: SynApps):
        result = sync_client.get_health()
        assert result["status"] == "healthy"
        assert "version" in result

    def test_get_health_detailed(self, sync_client: SynApps):
        result = sync_client.get_health_detailed()
        assert result["status"] in ("ok", "degraded", "down")

    def test_get_metrics(self, sync_client: SynApps):
        result = sync_client.get_metrics()
        assert isinstance(result, dict)

    def test_list_templates(self, sync_client: SynApps):
        result = sync_client.list_templates()
        assert isinstance(result, list)

    def test_list_flows(self, sync_client: SynApps):
        result = sync_client.list_flows()
        assert isinstance(result, dict)

    def test_create_and_get_flow(self, sync_client: SynApps):
        created = sync_client.create_flow(MINIMAL_FLOW)
        assert "id" in created
        fetched = sync_client.get_flow(created["id"])
        assert fetched["name"] == "SDK Test Flow"

    def test_delete_flow(self, sync_client: SynApps):
        created = sync_client.create_flow({**MINIMAL_FLOW, "name": "To Delete"})
        flow_id = created["id"]
        sync_client.delete_flow(flow_id)
        with pytest.raises(SynAppsAPIError) as exc_info:
            sync_client.get_flow(flow_id)
        assert exc_info.value.status_code == 404

    def test_export_flow(self, sync_client: SynApps):
        created = sync_client.create_flow({**MINIMAL_FLOW, "name": "Export Me"})
        exported = sync_client.export_flow(created["id"])
        assert exported["name"] == "Export Me"

    def test_list_providers(self, sync_client: SynApps):
        result = sync_client.list_providers()
        assert isinstance(result, list)

    def test_get_history(self, sync_client: SynApps):
        result = sync_client.get_history()
        assert isinstance(result, dict)

    def test_api_error_on_404(self, sync_client: SynApps):
        with pytest.raises(SynAppsAPIError) as exc_info:
            sync_client.get_flow("nonexistent-flow-id")
        assert exc_info.value.status_code == 404

    def test_api_error_fields(self, sync_client: SynApps):
        with pytest.raises(SynAppsAPIError) as exc_info:
            sync_client.get_task("nonexistent-task-id")
        err = exc_info.value
        assert err.status_code == 404
        assert isinstance(err.detail, str)
        assert isinstance(err.response_body, dict)

    def test_run_template_404(self, sync_client: SynApps):
        with pytest.raises(SynAppsAPIError) as exc_info:
            sync_client.run_template("nonexistent-template")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Async client tests
# ---------------------------------------------------------------------------

class TestAsyncClient:

    @pytest.mark.asyncio
    async def test_get_health(self, async_client: AsyncSynApps):
        result = await async_client.get_health()
        assert result["status"] == "healthy"
        assert "version" in result

    @pytest.mark.asyncio
    async def test_get_health_detailed(self, async_client: AsyncSynApps):
        result = await async_client.get_health_detailed()
        assert result["status"] in ("ok", "degraded", "down")

    @pytest.mark.asyncio
    async def test_get_metrics(self, async_client: AsyncSynApps):
        result = await async_client.get_metrics()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_templates(self, async_client: AsyncSynApps):
        result = await async_client.list_templates()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_create_and_get_flow(self, async_client: AsyncSynApps):
        created = await async_client.create_flow(MINIMAL_FLOW)
        assert "id" in created
        fetched = await async_client.get_flow(created["id"])
        assert fetched["name"] == "SDK Test Flow"

    @pytest.mark.asyncio
    async def test_delete_flow(self, async_client: AsyncSynApps):
        created = await async_client.create_flow({**MINIMAL_FLOW, "name": "Async Delete"})
        await async_client.delete_flow(created["id"])
        with pytest.raises(SynAppsAPIError) as exc_info:
            await async_client.get_flow(created["id"])
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_providers(self, async_client: AsyncSynApps):
        result = await async_client.list_providers()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_history(self, async_client: AsyncSynApps):
        result = await async_client.get_history()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_api_error_on_404(self, async_client: AsyncSynApps):
        with pytest.raises(SynAppsAPIError) as exc_info:
            await async_client.get_flow("no-such-flow")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_run_template_404(self, async_client: AsyncSynApps):
        with pytest.raises(SynAppsAPIError) as exc_info:
            await async_client.run_template("nonexistent-template")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Exception unit tests
# ---------------------------------------------------------------------------

class TestExceptions:

    def test_api_error_str(self):
        err = SynAppsAPIError(404, "Not found", {"detail": "Not found"})
        assert "404" in str(err)
        assert "Not found" in str(err)
        assert err.status_code == 404
        assert err.response_body == {"detail": "Not found"}

    def test_connection_error_is_synapps_error(self):
        from synapps_sdk.exceptions import SynAppsError
        assert isinstance(SynAppsConnectionError("refused"), SynAppsError)

    def test_timeout_error_is_synapps_error(self):
        from synapps_sdk.exceptions import SynAppsError
        assert isinstance(SynAppsTimeoutError("timeout"), SynAppsError)


# ---------------------------------------------------------------------------
# Poll task tests (mocked get_task)
# ---------------------------------------------------------------------------

class TestPollTask:

    def test_sync_poll_completed(self, sync_client: SynApps):
        call_count = 0
        def mock_get(tid: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"task_id": tid, "status": "running"}
            return {"task_id": tid, "status": "completed", "result": "done"}
        sync_client.get_task = mock_get  # type: ignore[assignment]
        result = sync_client.poll_task("t-1", poll_interval=0.01, timeout=5.0)
        assert result["status"] == "completed"
        assert call_count == 3

    def test_sync_poll_timeout(self, sync_client: SynApps):
        sync_client.get_task = lambda tid: {"task_id": tid, "status": "running"}  # type: ignore[assignment]
        with pytest.raises(SynAppsTimeoutError):
            sync_client.poll_task("t-stuck", poll_interval=0.01, timeout=0.05)

    def test_sync_poll_failed(self, sync_client: SynApps):
        sync_client.get_task = lambda tid: {"task_id": tid, "status": "failed", "error": "boom"}  # type: ignore[assignment]
        result = sync_client.poll_task("t-fail", poll_interval=0.01, timeout=5.0)
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_async_poll_completed(self, async_client: AsyncSynApps):
        call_count = 0
        async def mock_get(tid: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"task_id": tid, "status": "running"}
            return {"task_id": tid, "status": "completed", "result": "done"}
        async_client.get_task = mock_get  # type: ignore[assignment]
        result = await async_client.poll_task("t-1", poll_interval=0.01, timeout=5.0)
        assert result["status"] == "completed"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_poll_timeout(self, async_client: AsyncSynApps):
        async def always_running(tid: str) -> Dict[str, Any]:
            return {"task_id": tid, "status": "running"}
        async_client.get_task = always_running  # type: ignore[assignment]
        with pytest.raises(SynAppsTimeoutError):
            await async_client.poll_task("t-stuck", poll_interval=0.01, timeout=0.05)


# ---------------------------------------------------------------------------
# Client init tests
# ---------------------------------------------------------------------------

class TestClientInit:

    def test_default_base_url(self):
        c = SynApps()
        assert c.base_url == "http://localhost:8000/api/v1"
        c.close()

    def test_trailing_slash_stripped(self):
        c = SynApps(base_url="http://example.com/api/v1/")
        assert c.base_url == "http://example.com/api/v1"
        c.close()

    def test_api_key_in_headers(self):
        c = SynApps(api_key="sk-test")
        assert c._headers["X-API-Key"] == "sk-test"
        c.close()

    def test_no_api_key_no_header(self):
        c = SynApps()
        assert "X-API-Key" not in c._headers
        c.close()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:

    def test_version(self):
        import synapps_sdk
        assert synapps_sdk.__version__ == "0.1.0"

    def test_all_exports(self):
        import synapps_sdk
        for name in ("SynApps", "AsyncSynApps", "SynAppsError",
                     "SynAppsAPIError", "SynAppsConnectionError", "SynAppsTimeoutError"):
            assert hasattr(synapps_sdk, name)
