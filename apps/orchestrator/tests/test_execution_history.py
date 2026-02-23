"""Tests for workflow execution history + audit trail endpoints."""

import time
import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, HISTORY_VALID_STATUSES
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


async def _create_flow(name="Test Flow"):
    """Helper to create a flow and return its dict."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    return await FlowRepository.save({
        "name": name,
        "nodes": [
            {"id": f"start-{uid}", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": f"end-{uid}", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [{"id": f"e-{uid}", "source": f"start-{uid}", "target": f"end-{uid}"}],
    })


async def _create_run(flow_id, status="success", start_time=None, input_data=None):
    """Helper to create a run and return its dict."""
    import uuid
    return await WorkflowRunRepository.save({
        "run_id": str(uuid.uuid4()),
        "flow_id": flow_id,
        "status": status,
        "start_time": start_time or time.time(),
        "end_time": (start_time or time.time()) + 1.5 if status in ("success", "error") else None,
        "input_data": input_data or {"text": "hello"},
        "results": {},
        "total_steps": 2,
        "progress": 2 if status == "success" else 0,
    })


# ---------------------------------------------------------------------------
# GET /api/v1/history — list execution history
# ---------------------------------------------------------------------------


def test_history_returns_200(client):
    """GET /history returns 200 with expected shape."""
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


def test_history_empty(client):
    """GET /history with no runs returns empty list."""
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["history"] == []


@pytest.mark.asyncio
async def test_history_lists_runs(client):
    """GET /history lists created runs with enriched data."""
    flow = await _create_flow("My Workflow")
    await _create_run(flow["id"], status="success")
    await _create_run(flow["id"], status="error")

    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["history"]) == 2

    entry = data["history"][0]
    assert "run_id" in entry
    assert "flow_name" in entry
    assert "status" in entry
    assert "start_time" in entry
    assert "step_count" in entry
    assert "input_summary" in entry


@pytest.mark.asyncio
async def test_history_sorted_newest_first(client):
    """GET /history returns runs sorted by start_time descending."""
    flow = await _create_flow("Sorted Flow")
    await _create_run(flow["id"], status="success", start_time=1000.0)
    await _create_run(flow["id"], status="success", start_time=2000.0)
    await _create_run(flow["id"], status="success", start_time=1500.0)

    resp = client.get("/api/v1/history")
    data = resp.json()
    times = [e["start_time"] for e in data["history"]]
    assert times == sorted(times, reverse=True)


@pytest.mark.asyncio
async def test_history_entry_shape(client):
    """Each history entry has expected fields."""
    flow = await _create_flow("Shape Test")
    await _create_run(flow["id"], status="success", input_data={"url": "https://example.com"})

    resp = client.get("/api/v1/history")
    entry = resp.json()["history"][0]

    assert entry["flow_name"] == "Shape Test"
    assert entry["status"] == "success"
    assert entry["node_count"] == 2  # start + end
    assert isinstance(entry["steps_succeeded"], int)
    assert isinstance(entry["steps_failed"], int)
    assert entry["input_summary"] == {"url": "https://example.com"}
    assert entry["error"] is None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_filter_status(client):
    """GET /history?status=success filters correctly."""
    flow = await _create_flow("Filter Status")
    await _create_run(flow["id"], status="success")
    await _create_run(flow["id"], status="error")
    await _create_run(flow["id"], status="success")

    resp = client.get("/api/v1/history?status=success")
    data = resp.json()
    assert data["total"] == 2
    assert all(e["status"] == "success" for e in data["history"])


@pytest.mark.asyncio
async def test_history_filter_status_error(client):
    """GET /history?status=error returns only failed runs."""
    flow = await _create_flow("Filter Error")
    await _create_run(flow["id"], status="success")
    await _create_run(flow["id"], status="error")

    resp = client.get("/api/v1/history?status=error")
    data = resp.json()
    assert data["total"] == 1
    assert data["history"][0]["status"] == "error"


def test_history_filter_invalid_status(client):
    """GET /history?status=invalid returns 400."""
    resp = client.get("/api/v1/history?status=invalid")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_history_filter_template(client):
    """GET /history?template=... filters by flow name substring."""
    flow_a = await _create_flow("Content Engine")
    flow_b = await _create_flow("2Brain Triage")
    await _create_run(flow_a["id"], status="success")
    await _create_run(flow_b["id"], status="success")

    resp = client.get("/api/v1/history?template=content")
    data = resp.json()
    assert data["total"] == 1
    assert data["history"][0]["flow_name"] == "Content Engine"


@pytest.mark.asyncio
async def test_history_filter_date_range(client):
    """GET /history?start_after=...&start_before=... filters by date range."""
    flow = await _create_flow("Date Range")
    await _create_run(flow["id"], status="success", start_time=1000.0)
    await _create_run(flow["id"], status="success", start_time=2000.0)
    await _create_run(flow["id"], status="success", start_time=3000.0)

    resp = client.get("/api/v1/history?start_after=1500&start_before=2500")
    data = resp.json()
    assert data["total"] == 1
    assert data["history"][0]["start_time"] == 2000.0


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_pagination(client):
    """GET /history respects page and page_size."""
    flow = await _create_flow("Paginated")
    for i in range(5):
        await _create_run(flow["id"], status="success", start_time=1000.0 + i)

    resp = client.get("/api/v1/history?page=1&page_size=2")
    data = resp.json()
    assert data["total"] == 5
    assert len(data["history"]) == 2
    assert data["page"] == 1
    assert data["page_size"] == 2

    resp2 = client.get("/api/v1/history?page=3&page_size=2")
    data2 = resp2.json()
    assert data2["total"] == 5
    assert len(data2["history"]) == 1  # last page with 1 remaining


# ---------------------------------------------------------------------------
# GET /api/v1/history/{run_id} — execution detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_detail_returns_200(client):
    """GET /history/{run_id} returns full execution detail."""
    flow = await _create_flow("Detail Flow")
    run = await _create_run(flow["id"], status="success", input_data={"topic": "AI"})

    run_id = run["run_id"]
    resp = client.get(f"/api/v1/history/{run_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["run_id"] == run_id
    assert data["flow_name"] == "Detail Flow"
    assert data["status"] == "success"
    assert "trace" in data
    assert "input_data" in data
    assert data["input_data"] == {"topic": "AI"}


@pytest.mark.asyncio
async def test_history_detail_has_trace(client):
    """GET /history/{run_id} includes trace with nodes list."""
    flow = await _create_flow("Trace Flow")
    run = await _create_run(flow["id"], status="success")

    resp = client.get(f"/api/v1/history/{run['run_id']}")
    data = resp.json()
    trace = data["trace"]
    assert "run_id" in trace
    assert "status" in trace
    assert "nodes" in trace
    assert isinstance(trace["nodes"], list)


def test_history_detail_not_found(client):
    """GET /history/{run_id} returns 404 for unknown run."""
    resp = client.get("/api/v1/history/nonexistent-run-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_history_detail_error_run(client):
    """GET /history/{run_id} for a failed run includes error info."""
    flow = await _create_flow("Error Flow")
    run = await WorkflowRunRepository.save({
        "run_id": "err-run-123",
        "flow_id": flow["id"],
        "status": "error",
        "start_time": time.time(),
        "end_time": time.time() + 0.5,
        "input_data": {},
        "results": {},
        "error": "Node 'llm' timed out after 30s",
    })

    resp = client.get("/api/v1/history/err-run-123")
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"] == "Node 'llm' timed out after 30s"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_history_valid_statuses():
    """HISTORY_VALID_STATUSES contains expected values."""
    assert HISTORY_VALID_STATUSES == {"idle", "running", "success", "error"}
