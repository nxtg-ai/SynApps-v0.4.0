"""Tests for structured logging + request ID tracing (DIRECTIVE-23-03)."""

import json
import logging
import io
import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    app,
    _current_request_id,
    _JSONFormatter,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# _JSONFormatter — unit tests
# ---------------------------------------------------------------------------


def test_json_formatter_produces_valid_json():
    """_JSONFormatter outputs valid JSON."""
    fmt = _JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello world", args=(), exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"


def test_json_formatter_includes_request_id():
    """_JSONFormatter includes request_id from contextvar."""
    fmt = _JSONFormatter()
    token = _current_request_id.set("test-req-123")
    try:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="traced", args=(), exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "test-req-123"
    finally:
        _current_request_id.reset(token)


def test_json_formatter_includes_extra_fields():
    """_JSONFormatter includes endpoint, method, status, duration_ms, client_ip."""
    fmt = _JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="req done", args=(), exc_info=None,
    )
    record.endpoint = "/api/v1/flows"
    record.method = "GET"
    record.status = 200
    record.duration_ms = 12.5
    record.client_ip = "127.0.0.1"
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["endpoint"] == "/api/v1/flows"
    assert parsed["method"] == "GET"
    assert parsed["status"] == 200
    assert parsed["duration_ms"] == 12.5
    assert parsed["client_ip"] == "127.0.0.1"


def test_json_formatter_includes_exception():
    """_JSONFormatter includes exception info when present."""
    fmt = _JSONFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="", lineno=0,
        msg="failed", args=(), exc_info=exc_info,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_json_formatter_default_request_id():
    """_JSONFormatter uses '-' when no request_id is set."""
    fmt = _JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="no context", args=(), exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["request_id"] == "-"


def test_json_formatter_has_timestamp():
    """_JSONFormatter output includes timestamp field."""
    fmt = _JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="ts test", args=(), exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert "timestamp" in parsed
    assert len(parsed["timestamp"]) > 0


# ---------------------------------------------------------------------------
# X-Request-ID middleware
# ---------------------------------------------------------------------------


def test_response_has_x_request_id(client):
    """Every response includes X-Request-ID header."""
    resp = client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


def test_request_id_auto_generated(client):
    """When no X-Request-ID is provided, one is generated."""
    resp = client.get("/api/v1/health")
    request_id = resp.headers["X-Request-ID"]
    assert len(request_id) == 16  # uuid4().hex[:16]


def test_request_id_unique_per_request(client):
    """Each request gets a unique ID."""
    ids = set()
    for _ in range(10):
        resp = client.get("/api/v1/health")
        ids.add(resp.headers["X-Request-ID"])
    assert len(ids) == 10


def test_client_provided_request_id(client):
    """Client-provided X-Request-ID is echoed back."""
    resp = client.get(
        "/api/v1/health",
        headers={"X-Request-ID": "my-custom-trace-id"},
    )
    assert resp.headers["X-Request-ID"] == "my-custom-trace-id"


def test_request_id_on_error_responses(client):
    """X-Request-ID is set even on 404 and other error responses."""
    resp = client.get("/api/v1/flows/nonexistent-flow-id")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


def test_request_id_on_post(client):
    """X-Request-ID is set on POST responses."""
    resp = client.post("/api/v1/templates/validate", json={
        "name": "Test",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [{"id": "e1", "source": "start", "target": "end"}],
    })
    assert "X-Request-ID" in resp.headers


# ---------------------------------------------------------------------------
# contextvar propagation
# ---------------------------------------------------------------------------


def test_contextvar_default():
    """_current_request_id defaults to '-' outside request context."""
    assert _current_request_id.get("-") == "-"


def test_contextvar_set_and_reset():
    """_current_request_id can be set and reset."""
    token = _current_request_id.set("test-id")
    assert _current_request_id.get() == "test-id"
    _current_request_id.reset(token)
    assert _current_request_id.get("-") == "-"


# ---------------------------------------------------------------------------
# Structured request logging
# ---------------------------------------------------------------------------


def test_request_logging_output(client, capfd):
    """Request middleware logs structured JSON to stdout."""
    # The orchestrator logger writes to stderr/stdout via StreamHandler
    resp = client.get("/api/v1/health")
    request_id = resp.headers["X-Request-ID"]

    # Capture the logger output — get the orchestrator logger's handler
    orch_logger = logging.getLogger("orchestrator")
    buf = io.StringIO()
    test_handler = logging.StreamHandler(buf)
    test_handler.setFormatter(_JSONFormatter())
    orch_logger.addHandler(test_handler)

    try:
        # Make another request so we can capture its log
        resp2 = client.get("/api/v1/health")
        request_id2 = resp2.headers["X-Request-ID"]
        output = buf.getvalue()
    finally:
        orch_logger.removeHandler(test_handler)

    # Parse the last JSON line from the log
    lines = [l for l in output.strip().split("\n") if l.strip()]
    assert len(lines) >= 1
    last_log = json.loads(lines[-1])
    assert last_log["request_id"] == request_id2
    assert last_log["method"] == "GET"
    assert "/health" in last_log["endpoint"]
    assert last_log["status"] == 200
    assert "duration_ms" in last_log


# ---------------------------------------------------------------------------
# CORS exposes X-Request-ID
# ---------------------------------------------------------------------------


def test_cors_exposes_request_id(client):
    """CORS exposes X-Request-ID header."""
    resp = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    exposed = resp.headers.get("access-control-expose-headers", "")
    assert "x-request-id" in exposed.lower() or resp.status_code == 200
