"""Tests for async task queue + background execution endpoints."""

import time
import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, task_queue, _load_yaml_template


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_tasks():
    """Clean task queue between tests."""
    task_queue.reset()
    yield
    task_queue.reset()


# ---------------------------------------------------------------------------
# TaskQueue — unit tests
# ---------------------------------------------------------------------------


def test_task_create():
    """create() returns a task ID and sets pending status."""
    task_id = task_queue.create("test-template", "Test Flow")
    assert task_id
    task = task_queue.get(task_id)
    assert task["status"] == "pending"
    assert task["template_id"] == "test-template"
    assert task["flow_name"] == "Test Flow"
    assert task["progress_pct"] == 0
    assert task["run_id"] is None
    assert task["result"] is None
    assert task["error"] is None


def test_task_update():
    """update() modifies task fields."""
    task_id = task_queue.create("t", "F")
    task_queue.update(task_id, status="running", progress_pct=50)
    task = task_queue.get(task_id)
    assert task["status"] == "running"
    assert task["progress_pct"] == 50


def test_task_get_nonexistent():
    """get() returns None for unknown task ID."""
    assert task_queue.get("nonexistent") is None


def test_task_list_all():
    """list_tasks() returns all tasks sorted by created_at desc."""
    task_queue.create("a", "A")
    task_queue.create("b", "B")
    tasks = task_queue.list_tasks()
    assert len(tasks) == 2
    # Most recent first
    assert tasks[0]["template_id"] == "b"


def test_task_list_filter_status():
    """list_tasks(status=...) filters by status."""
    t1 = task_queue.create("a", "A")
    task_queue.create("b", "B")
    task_queue.update(t1, status="completed")
    completed = task_queue.list_tasks(status="completed")
    assert len(completed) == 1
    assert completed[0]["task_id"] == t1
    pending = task_queue.list_tasks(status="pending")
    assert len(pending) == 1


def test_task_reset():
    """reset() clears all tasks."""
    task_queue.create("a", "A")
    task_queue.reset()
    assert task_queue.list_tasks() == []


# ---------------------------------------------------------------------------
# _load_yaml_template
# ---------------------------------------------------------------------------


def test_load_yaml_template_found():
    """Loading an existing template by ID returns its data."""
    data = _load_yaml_template("content-engine-pipeline")
    if data is None:
        pytest.skip("content_engine.yaml not found in templates/")
    assert data["name"] == "Content Engine Pipeline"
    assert "nodes" in data
    assert "edges" in data


def test_load_yaml_template_not_found():
    """Loading a non-existent template returns None."""
    assert _load_yaml_template("nonexistent-template-xyz") is None


# ---------------------------------------------------------------------------
# API: POST /templates/{id}/run-async
# ---------------------------------------------------------------------------


def test_run_async_returns_202(client):
    """POST /templates/{id}/run-async returns 202 with task_id."""
    data = _load_yaml_template("content-engine-pipeline")
    if data is None:
        pytest.skip("content_engine.yaml not found")
    resp = client.post(
        "/api/v1/templates/content-engine-pipeline/run-async",
        json={"input": {"url": "https://example.com", "topic": "Test"}},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body
    assert body["status"] == "pending"


def test_run_async_unknown_template(client):
    """POST /templates/{id}/run-async returns 404 for unknown template."""
    resp = client.post(
        "/api/v1/templates/nonexistent-xyz/run-async",
        json={"input": {}},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API: GET /tasks/{id}
# ---------------------------------------------------------------------------


def test_get_task_endpoint(client):
    """GET /tasks/{id} returns task data after creating via run-async."""
    data = _load_yaml_template("content-engine-pipeline")
    if data is None:
        pytest.skip("content_engine.yaml not found")
    create = client.post(
        "/api/v1/templates/content-engine-pipeline/run-async",
        json={"input": {"url": "https://example.com"}},
    )
    task_id = create.json()["task_id"]
    resp = client.get(f"/api/v1/tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["template_id"] == "content-engine-pipeline"
    assert body["status"] in ("pending", "running", "completed", "failed")
    assert "progress_pct" in body


def test_get_task_not_found(client):
    """GET /tasks/{id} returns 404 for unknown task."""
    resp = client.get("/api/v1/tasks/nonexistent-task-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API: GET /tasks
# ---------------------------------------------------------------------------


def test_list_tasks_endpoint(client):
    """GET /tasks returns all tasks."""
    # Create a couple tasks directly
    task_queue.create("t1", "Flow 1")
    task_queue.create("t2", "Flow 2")
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["tasks"]) == 2


def test_list_tasks_with_status_filter(client):
    """GET /tasks?status=pending filters correctly."""
    t1 = task_queue.create("t1", "F1")
    task_queue.create("t2", "F2")
    task_queue.update(t1, status="completed")
    resp = client.get("/api/v1/tasks?status=pending")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["tasks"][0]["status"] == "pending"


def test_list_tasks_invalid_status_filter(client):
    """GET /tasks?status=invalid returns 400."""
    resp = client.get("/api/v1/tasks?status=invalid")
    assert resp.status_code == 400


def test_list_tasks_empty(client):
    """GET /tasks with no tasks returns empty list."""
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
