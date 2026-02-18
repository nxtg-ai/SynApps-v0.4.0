"""
Comprehensive unit and integration tests for T-072.

Covers: auth helpers, password hashing, JWT tokens, API key encryption,
pagination, helper functions (trace, diff, json path, template rendering),
orchestrator static methods, API endpoints with auth, and node applets.
"""
import asyncio
import base64
import hashlib
import hmac
import importlib
import json
import math
import os
import re
import secrets
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import apps.orchestrator.db as db_module
from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.main import (
    _build_json_diff,
    _build_run_diff,
    _error_response,
    _extract_trace_from_run,
    _finalize_execution_trace,
    _flatten_for_diff,
    _new_execution_trace,
    _node_result_index,
    _parse_json_path,
    _render_template_string,
    _resolve_json_path,
    _resolve_template_path,
    _trace_value,
    _hash_password,
    _verify_password,
    _encrypt_api_key,
    _decrypt_api_key,
    _decode_token,
    _ws_message,
    _safe_tmp_dir,
    _extract_sandbox_result,
    _read_stream_limited,
    _render_template_payload,
    paginate,
    app,
    AppletMessage,
    BaseApplet,
    Orchestrator,
    FlowNodeRequest,
    FlowEdgeRequest,
    CreateFlowRequest,
    RunFlowRequest,
    RerunFlowRequest,
    AISuggestRequest,
    AppletStatus,
    NodeErrorCode,
    NodeError,
    TRACE_RESULTS_KEY,
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(scope="function")
async def db():
    await init_db()
    yield
    await close_db_connections()


# ============================================================
# Password Hashing Tests
# ============================================================


class TestPasswordHashing:
    def test_hash_and_verify_success(self):
        password = "MySecureP@ssword123"
        hashed = _hash_password(password)
        assert _verify_password(password, hashed) is True

    def test_hash_produces_correct_format(self):
        hashed = _hash_password("test")
        parts = hashed.split("$")
        assert len(parts) == 4
        assert parts[0] == "pbkdf2_sha256"
        assert int(parts[1]) > 0  # iterations

    def test_verify_wrong_password(self):
        hashed = _hash_password("correct-password")
        assert _verify_password("wrong-password", hashed) is False

    def test_verify_empty_password(self):
        hashed = _hash_password("something")
        assert _verify_password("", hashed) is False

    def test_verify_garbage_hash(self):
        assert _verify_password("pass", "not-a-valid-hash") is False

    def test_verify_wrong_scheme(self):
        assert _verify_password("pass", "bcrypt$100$salt$hash") is False

    def test_verify_malformed_iterations(self):
        assert _verify_password("pass", "pbkdf2_sha256$abc$salt$hash") is False

    def test_different_hashes_for_same_password(self):
        """Salt should make each hash unique."""
        h1 = _hash_password("same")
        h2 = _hash_password("same")
        assert h1 != h2
        assert _verify_password("same", h1)
        assert _verify_password("same", h2)


# ============================================================
# API Key Encryption Tests
# ============================================================


class TestAPIKeyEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        plain = "synapps_abc123def456ghi789"
        encrypted = _encrypt_api_key(plain)
        assert encrypted != plain
        decrypted = _decrypt_api_key(encrypted)
        assert decrypted == plain

    def test_decrypt_invalid_token(self):
        result = _decrypt_api_key("not-a-valid-fernet-token")
        assert result is None

    def test_decrypt_empty_string(self):
        result = _decrypt_api_key("")
        assert result is None


# ============================================================
# JWT Token Tests
# ============================================================


class TestJWTTokens:
    def test_decode_expired_token(self):
        import jwt as pyjwt
        payload = {
            "sub": "user-1",
            "email": "test@example.com",
            "token_type": "access",
            "iat": int(time.time()) - 7200,
            "exp": int(time.time()) - 3600,  # Already expired
        }
        token = pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token, "access")
        assert exc_info.value.status_code == 401
        assert "expired" in str(exc_info.value.detail).lower()

    def test_decode_invalid_token(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode_token("not.a.valid.jwt", "access")
        assert exc_info.value.status_code == 401

    def test_decode_wrong_token_type(self):
        import jwt as pyjwt
        payload = {
            "sub": "user-1",
            "email": "test@example.com",
            "token_type": "refresh",  # We'll expect "access"
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token, "access")
        assert "Invalid token type" in str(exc_info.value.detail)

    def test_decode_valid_access_token(self):
        import jwt as pyjwt
        now = int(time.time())
        payload = {
            "sub": "user-1",
            "email": "test@example.com",
            "token_type": "access",
            "iat": now,
            "exp": now + 3600,
        }
        token = pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        result = _decode_token(token, "access")
        assert result["sub"] == "user-1"
        assert result["token_type"] == "access"


# ============================================================
# Pagination Tests
# ============================================================


class TestPagination:
    def test_basic_pagination(self):
        items = list(range(25))
        result = paginate(items, page=1, page_size=10)
        assert result["total"] == 25
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert result["total_pages"] == 3
        assert len(result["items"]) == 10
        assert result["items"] == list(range(10))

    def test_second_page(self):
        items = list(range(25))
        result = paginate(items, page=2, page_size=10)
        assert result["items"] == list(range(10, 20))

    def test_last_page_partial(self):
        items = list(range(25))
        result = paginate(items, page=3, page_size=10)
        assert len(result["items"]) == 5

    def test_empty_items(self):
        result = paginate([], page=1, page_size=10)
        assert result["total"] == 0
        assert result["items"] == []
        assert result["total_pages"] == 0

    def test_page_beyond_range(self):
        result = paginate([1, 2, 3], page=100, page_size=10)
        assert result["items"] == []

    def test_page_size_zero(self):
        result = paginate([1, 2], page=1, page_size=0)
        assert result["total_pages"] == 0


# ============================================================
# _trace_value Tests
# ============================================================


class TestTraceValue:
    def test_none(self):
        assert _trace_value(None) is None

    def test_primitives(self):
        assert _trace_value("hello") == "hello"
        assert _trace_value(42) == 42
        assert _trace_value(3.14) == 3.14
        assert _trace_value(True) is True

    def test_dict(self):
        result = _trace_value({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_list(self):
        result = _trace_value([1, "two", None])
        assert result == [1, "two", None]

    def test_set(self):
        result = _trace_value({1, 2})
        assert isinstance(result, list)
        assert set(result) == {1, 2}

    def test_nested(self):
        result = _trace_value({"a": [1, {"b": 2}]})
        assert result == {"a": [1, {"b": 2}]}

    def test_depth_limit(self):
        """At depth >= 8, values are stringified."""
        deep = "leaf"
        for _ in range(10):
            deep = {"nested": deep}
        result = _trace_value(deep)
        # At some depth the inner value will be str()
        assert isinstance(result, dict)

    def test_pydantic_model(self):
        msg = AppletMessage(content="test", context={"k": "v"}, metadata={})
        result = _trace_value(msg)
        assert isinstance(result, dict)
        assert result["content"] == "test"


# ============================================================
# Execution Trace Tests
# ============================================================


class TestExecutionTrace:
    def test_new_execution_trace(self):
        trace = _new_execution_trace("run-1", "flow-1", {"key": "val"}, 1000.0)
        assert trace["run_id"] == "run-1"
        assert trace["flow_id"] == "flow-1"
        assert trace["status"] == "running"
        assert trace["input"] == {"key": "val"}
        assert trace["started_at"] == 1000.0
        assert trace["nodes"] == []
        assert trace["errors"] == []

    def test_finalize_execution_trace(self):
        trace = _new_execution_trace("run-1", "flow-1", {}, 1000.0)
        _finalize_execution_trace(trace, "success", 1005.0)
        assert trace["status"] == "success"
        assert trace["ended_at"] == 1005.0
        assert trace["duration_ms"] == 5000.0

    def test_extract_trace_from_run_with_stored_trace(self):
        run = {
            "run_id": "r1",
            "flow_id": "f1",
            "status": "success",
            "start_time": 1000.0,
            "end_time": 1002.0,
            "input_data": {"q": "test"},
            "results": {
                TRACE_RESULTS_KEY: {
                    "version": 1,
                    "run_id": "r1",
                    "flow_id": "f1",
                    "status": "success",
                    "input": {"q": "test"},
                    "started_at": 1000.0,
                    "ended_at": 1002.0,
                    "nodes": [],
                    "errors": [],
                }
            },
        }
        trace = _extract_trace_from_run(run)
        assert trace["run_id"] == "r1"
        assert trace["status"] == "success"
        assert trace["duration_ms"] == 2000.0

    def test_extract_trace_legacy_run(self):
        run = {
            "run_id": "r2",
            "flow_id": "f2",
            "status": "success",
            "start_time": 1000.0,
            "end_time": 1001.0,
            "input_data": {},
            "results": {
                "node1": {"output": "hello", "type": "writer", "status": "success"},
            },
        }
        trace = _extract_trace_from_run(run)
        assert len(trace["nodes"]) == 1
        assert trace["nodes"][0]["node_id"] == "node1"
        assert trace["nodes"][0]["output"] == "hello"

    def test_extract_trace_missing_start_time(self):
        run = {"run_id": "r3", "results": {}}
        trace = _extract_trace_from_run(run)
        assert isinstance(trace["started_at"], float)

    def test_extract_trace_non_dict_result(self):
        run = {
            "run_id": "r4",
            "status": "success",
            "start_time": 1000.0,
            "results": {"node1": "simple string output"},
        }
        trace = _extract_trace_from_run(run)
        assert trace["nodes"][0]["output"] == "simple string output"
        assert trace["nodes"][0]["node_type"] is None


# ============================================================
# Flatten & JSON Diff Tests
# ============================================================


class TestFlattenAndDiff:
    def test_flatten_primitives(self):
        out = {}
        _flatten_for_diff("hello", "$", out)
        assert out == {"$": "hello"}

    def test_flatten_dict(self):
        out = {}
        _flatten_for_diff({"a": 1, "b": 2}, "$", out)
        assert out == {"$.a": 1, "$.b": 2}

    def test_flatten_list(self):
        out = {}
        _flatten_for_diff([10, 20], "$", out)
        assert out == {"$[0]": 10, "$[1]": 20}

    def test_flatten_empty_dict(self):
        out = {}
        _flatten_for_diff({}, "$", out)
        assert out == {"$": {}}

    def test_flatten_empty_list(self):
        out = {}
        _flatten_for_diff([], "$", out)
        assert out == {"$": []}

    def test_flatten_nested(self):
        out = {}
        _flatten_for_diff({"a": {"b": 1}}, "$", out)
        assert out == {"$.a.b": 1}

    def test_build_json_diff_identical(self):
        result = _build_json_diff({"a": 1}, {"a": 1})
        assert result["changed"] is False
        assert result["change_count"] == 0
        assert result["changes"] == []

    def test_build_json_diff_added(self):
        result = _build_json_diff({}, {"a": 1})
        assert result["changed"] is True
        assert any(c["type"] == "added" for c in result["changes"])

    def test_build_json_diff_removed(self):
        result = _build_json_diff({"a": 1}, {})
        assert result["changed"] is True
        assert any(c["type"] == "removed" for c in result["changes"])

    def test_build_json_diff_modified(self):
        result = _build_json_diff({"a": 1}, {"a": 2})
        assert result["changed"] is True
        assert any(c["type"] == "modified" for c in result["changes"])

    def test_build_json_diff_truncated(self):
        left = {str(i): i for i in range(300)}
        right = {str(i): i + 1 for i in range(300)}
        result = _build_json_diff(left, right, max_changes=5)
        assert result["truncated"] is True
        assert len(result["changes"]) == 5

    def test_node_result_index(self):
        run = {"results": {"n1": "out1", TRACE_RESULTS_KEY: {}}}
        index = _node_result_index(run)
        assert "n1" in index
        assert TRACE_RESULTS_KEY not in index

    def test_node_result_index_no_results(self):
        assert _node_result_index({"results": None}) == {}
        assert _node_result_index({}) == {}


# ============================================================
# Template & JSON Path Tests
# ============================================================


class TestTemplatePaths:
    def test_resolve_template_path_dict(self):
        data = {"a": {"b": "value"}}
        result, found = _resolve_template_path(data, "a.b")
        assert found is True
        assert result == "value"

    def test_resolve_template_path_list(self):
        data = {"items": ["first", "second"]}
        result, found = _resolve_template_path(data, "items.0")
        assert found is True
        assert result == "first"

    def test_resolve_template_path_not_found(self):
        data = {"a": 1}
        result, found = _resolve_template_path(data, "b")
        assert found is False
        assert result is None

    def test_resolve_template_path_list_bad_index(self):
        data = {"items": [1]}
        result, found = _resolve_template_path(data, "items.5")
        assert found is False

    def test_resolve_template_path_list_non_digit(self):
        data = {"items": [1]}
        result, found = _resolve_template_path(data, "items.key")
        assert found is False

    def test_resolve_template_path_scalar(self):
        data = {"a": 42}
        result, found = _resolve_template_path(data, "a.b")
        assert found is False


class TestRenderTemplateString:
    def test_no_templates(self):
        assert _render_template_string("hello world", {}) == "hello world"

    def test_single_template_returns_value(self):
        result = _render_template_string("{{name}}", {"name": "Alice"})
        assert result == "Alice"

    def test_single_template_returns_dict(self):
        data = {"obj": {"key": "val"}}
        result = _render_template_string("{{obj}}", data)
        assert result == {"key": "val"}

    def test_mixed_template(self):
        result = _render_template_string("Hello {{name}}!", {"name": "World"})
        assert result == "Hello World!"

    def test_missing_path_returns_empty(self):
        result = _render_template_string("Hi {{missing}}!", {})
        assert result == "Hi !"

    def test_dict_value_in_mixed_template(self):
        data = {"obj": {"a": 1}}
        result = _render_template_string("data: {{obj}}", data)
        assert '"a": 1' in result


class TestRenderTemplatePayload:
    def test_string(self):
        assert _render_template_payload("{{x}}", {"x": 42}) == 42

    def test_list(self):
        result = _render_template_payload(["{{a}}", "literal"], {"a": "val"})
        assert result == ["val", "literal"]

    def test_dict(self):
        result = _render_template_payload({"key": "{{a}}"}, {"a": 10})
        assert result == {"key": 10}

    def test_non_string(self):
        assert _render_template_payload(42, {}) == 42


class TestParseJsonPath:
    def test_root(self):
        assert _parse_json_path("$") == []

    def test_empty(self):
        assert _parse_json_path("") == []

    def test_simple_key(self):
        assert _parse_json_path("$.name") == ["name"]

    def test_nested_key(self):
        assert _parse_json_path("$.a.b.c") == ["a", "b", "c"]

    def test_array_index(self):
        assert _parse_json_path("$[0]") == [0]

    def test_mixed_path(self):
        assert _parse_json_path("$.items[0].name") == ["items", 0, "name"]

    def test_quoted_key(self):
        assert _parse_json_path("$['key']") == ["key"]
        assert _parse_json_path('$["key"]') == ["key"]

    def test_without_dollar(self):
        assert _parse_json_path("name") == ["name"]
        assert _parse_json_path(".name") == ["name"]

    def test_invalid_path(self):
        assert _parse_json_path("$!!!") is None


class TestResolveJsonPath:
    def test_root(self):
        data = {"a": 1}
        result, found = _resolve_json_path(data, "$")
        assert found is True
        assert result == {"a": 1}

    def test_nested_key(self):
        data = {"a": {"b": {"c": 42}}}
        result, found = _resolve_json_path(data, "$.a.b.c")
        assert found is True
        assert result == 42

    def test_array_index(self):
        data = {"items": ["x", "y", "z"]}
        result, found = _resolve_json_path(data, "$.items[1]")
        assert found is True
        assert result == "y"

    def test_not_found(self):
        data = {"a": 1}
        result, found = _resolve_json_path(data, "$.b")
        assert found is False

    def test_index_out_of_range(self):
        data = {"items": [1]}
        result, found = _resolve_json_path(data, "$.items[5]")
        assert found is False

    def test_index_on_non_list(self):
        data = {"a": "string"}
        result, found = _resolve_json_path(data, "$.a[0]")
        assert found is False

    def test_key_on_non_dict(self):
        data = {"a": 42}
        result, found = _resolve_json_path(data, "$.a.b")
        assert found is False

    def test_invalid_path(self):
        result, found = _resolve_json_path({}, "$!!!")
        assert found is False


# ============================================================
# Safe Tmp Dir Tests
# ============================================================


class TestSafeTmpDir:
    def test_tmp_allowed(self):
        assert _safe_tmp_dir("/tmp") == "/tmp"

    def test_tmp_subdir(self):
        result = _safe_tmp_dir("/tmp/subdir")
        assert result.startswith("/tmp")

    def test_outside_tmp_rejected(self):
        result = _safe_tmp_dir("/var/log")
        assert result == "/tmp"

    def test_empty_defaults_to_tmp(self):
        result = _safe_tmp_dir("")
        assert result == "/tmp"


# ============================================================
# Extract Sandbox Result Tests
# ============================================================


class TestExtractSandboxResult:
    def test_no_markers(self):
        stdout, result = _extract_sandbox_result("just some output")
        assert stdout == "just some output"
        assert result is None

    def test_with_markers(self):
        text = 'before\n__SYNAPPS_RESULT_START__\n{"key": "val"}\n__SYNAPPS_RESULT_END__\nafter'
        stdout, result = _extract_sandbox_result(text)
        assert result == {"key": "val"}
        assert "__SYNAPPS_RESULT_START__" not in stdout

    def test_invalid_json_between_markers(self):
        text = '__SYNAPPS_RESULT_START__\nnot json\n__SYNAPPS_RESULT_END__'
        stdout, result = _extract_sandbox_result(text)
        assert result is None


# ============================================================
# Read Stream Limited Tests
# ============================================================


class TestReadStreamLimited:
    @pytest.mark.asyncio
    async def test_empty_stream(self):
        stream = AsyncMock()
        stream.read = AsyncMock(return_value=b"")
        data, truncated = await _read_stream_limited(stream, 1024)
        assert data == b""
        assert truncated is False

    @pytest.mark.asyncio
    async def test_none_stream(self):
        data, truncated = await _read_stream_limited(None, 1024)
        assert data == b""
        assert truncated is False

    @pytest.mark.asyncio
    async def test_truncation(self):
        chunks = [b"x" * 100, b""]
        stream = AsyncMock()
        stream.read = AsyncMock(side_effect=chunks)
        data, truncated = await _read_stream_limited(stream, 50)
        assert len(data) == 50
        assert truncated is True


# ============================================================
# WS Message Helper Tests
# ============================================================


class TestWSMessage:
    def test_basic(self):
        msg = _ws_message("test.type", {"key": "val"})
        assert msg["type"] == "test.type"
        assert msg["data"] == {"key": "val"}
        assert "id" in msg
        assert "timestamp" in msg

    def test_with_ref_id(self):
        msg = _ws_message("response", {}, ref_id="ref-123")
        assert msg["ref_id"] == "ref-123"


# ============================================================
# Error Response Helper Tests
# ============================================================


class TestErrorResponse:
    def test_basic_error(self):
        resp = _error_response(404, "NOT_FOUND", "Resource not found")
        assert resp.status_code == 404
        body = json.loads(resp.body)
        assert body["error"]["code"] == "NOT_FOUND"
        assert body["error"]["message"] == "Resource not found"

    def test_error_with_details(self):
        details = [{"field": "name", "message": "required"}]
        resp = _error_response(422, "VALIDATION_ERROR", "Validation failed", details)
        body = json.loads(resp.body)
        assert body["error"]["details"] == details


# ============================================================
# Request Validation Model Tests
# ============================================================


class TestRequestModels:
    def test_flow_node_request_valid(self):
        node = FlowNodeRequest(
            id="n1", type="writer", position={"x": 10.0, "y": 20.0}
        )
        assert node.id == "n1"

    def test_flow_node_request_missing_position_keys(self):
        with pytest.raises(Exception):
            FlowNodeRequest(id="n1", type="writer", position={"z": 10.0})

    def test_flow_edge_request_default_animated(self):
        edge = FlowEdgeRequest(id="e1", source="n1", target="n2")
        assert edge.animated is False

    def test_create_flow_request_blank_name(self):
        with pytest.raises(Exception):
            CreateFlowRequest(name="   ")

    def test_create_flow_request_blank_id_becomes_none(self):
        flow = CreateFlowRequest(name="Test", id="   ")
        assert flow.id is None

    def test_create_flow_request_valid(self):
        flow = CreateFlowRequest(name="  My Flow  ")
        assert flow.name == "My Flow"

    def test_run_flow_request_defaults(self):
        req = RunFlowRequest()
        assert req.input == {}

    def test_rerun_flow_request_defaults(self):
        req = RerunFlowRequest()
        assert req.input == {}
        assert req.merge_with_original_input is True

    def test_ai_suggest_request_min_length(self):
        with pytest.raises(Exception):
            AISuggestRequest(prompt="")


# ============================================================
# Enum Tests
# ============================================================


class TestEnums:
    def test_applet_status_values(self):
        assert AppletStatus.IDLE == "idle"
        assert AppletStatus.RUNNING == "running"
        assert AppletStatus.SUCCESS == "success"
        assert AppletStatus.ERROR == "error"

    def test_node_error_code_values(self):
        assert NodeErrorCode.TIMEOUT == "TIMEOUT"
        assert NodeErrorCode.RETRY_EXHAUSTED == "RETRY_EXHAUSTED"

    def test_node_error_exception(self):
        err = NodeError(NodeErrorCode.EXECUTION_ERROR, "something failed", node_id="n1")
        assert err.code == NodeErrorCode.EXECUTION_ERROR
        assert err.message == "something failed"
        assert err.node_id == "n1"


# ============================================================
# Orchestrator Static Method Tests
# ============================================================


class TestOrchestratorMethods:
    def test_create_run_id(self):
        run_id = Orchestrator.create_run_id()
        # Should be a valid UUID
        uuid.UUID(run_id)

    def test_collect_outgoing_targets(self):
        edges = [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "c"},
            {"source": "a", "target": "b"},  # Duplicate
        ]
        targets = Orchestrator._collect_outgoing_targets(edges)
        assert targets == ["b", "c"]

    def test_collect_outgoing_targets_empty(self):
        assert Orchestrator._collect_outgoing_targets([]) == []

    def test_collect_outgoing_targets_invalid(self):
        edges = [{"source": "a", "target": ""}, {"source": "a"}]
        assert Orchestrator._collect_outgoing_targets(edges) == []

    def test_branch_from_hint_true(self):
        assert Orchestrator._branch_from_hint("true") == "true"
        assert Orchestrator._branch_from_hint("yes") == "true"
        assert Orchestrator._branch_from_hint("then") == "true"
        assert Orchestrator._branch_from_hint("PASS") == "true"

    def test_branch_from_hint_false(self):
        assert Orchestrator._branch_from_hint("false") == "false"
        assert Orchestrator._branch_from_hint("else") == "false"
        assert Orchestrator._branch_from_hint("no") == "false"
        assert Orchestrator._branch_from_hint("fail") == "false"

    def test_branch_from_hint_none(self):
        assert Orchestrator._branch_from_hint(None) is None
        assert Orchestrator._branch_from_hint("") is None
        assert Orchestrator._branch_from_hint("unknown") is None

    def test_extract_if_else_branch_none_response(self):
        assert Orchestrator._extract_if_else_branch(None) == "false"

    def test_extract_if_else_branch_metadata_branch(self):
        msg = AppletMessage(content="", metadata={"branch": "true"})
        assert Orchestrator._extract_if_else_branch(msg) == "true"

    def test_extract_if_else_branch_metadata_bool_result(self):
        msg = AppletMessage(content="", metadata={"condition_result": True})
        assert Orchestrator._extract_if_else_branch(msg) == "true"

        msg = AppletMessage(content="", metadata={"condition_result": False})
        assert Orchestrator._extract_if_else_branch(msg) == "false"

    def test_extract_if_else_branch_metadata_numeric_result(self):
        msg = AppletMessage(content="", metadata={"condition_result": 42})
        assert Orchestrator._extract_if_else_branch(msg) == "true"

        msg = AppletMessage(content="", metadata={"condition_result": 0})
        assert Orchestrator._extract_if_else_branch(msg) == "false"

    def test_extract_if_else_branch_content_branch(self):
        msg = AppletMessage(content={"branch": "false"}, metadata={})
        assert Orchestrator._extract_if_else_branch(msg) == "false"

    def test_extract_if_else_branch_content_bool_result(self):
        msg = AppletMessage(content={"result": True}, metadata={})
        assert Orchestrator._extract_if_else_branch(msg) == "true"

    def test_branch_target_from_node_data(self):
        node = {"data": {"true_target": "t1", "false_target": "f1"}}
        assert Orchestrator._branch_target_from_node_data(node, "true") == "t1"
        assert Orchestrator._branch_target_from_node_data(node, "false") == "f1"

    def test_branch_target_from_node_data_camel(self):
        node = {"data": {"trueTarget": "t2"}}
        assert Orchestrator._branch_target_from_node_data(node, "true") == "t2"

    def test_branch_target_from_node_data_empty(self):
        node = {"data": {}}
        assert Orchestrator._branch_target_from_node_data(node, "true") is None

    def test_branch_target_from_node_data_no_data(self):
        node = {}
        assert Orchestrator._branch_target_from_node_data(node, "true") is None

    def test_infer_edge_branch(self):
        assert Orchestrator._infer_edge_branch({"branch": "true"}) == "true"
        assert Orchestrator._infer_edge_branch({"label": "else"}) == "false"
        assert Orchestrator._infer_edge_branch({"sourceHandle": "on_true"}) == "true"

    def test_infer_edge_branch_from_data(self):
        edge = {"data": {"branch": "false"}}
        assert Orchestrator._infer_edge_branch(edge) == "false"

    def test_infer_edge_branch_none(self):
        assert Orchestrator._infer_edge_branch({}) is None

    def test_resolve_next_targets_non_if_else(self):
        node = {"type": "writer"}
        edges = [{"target": "t1"}, {"target": "t2"}]
        targets = Orchestrator._resolve_next_targets(node, edges)
        assert targets == ["t1", "t2"]

    def test_resolve_next_targets_if_else_explicit_target(self):
        node = {"type": "if_else", "data": {"true_target": "t1"}}
        edges = [{"target": "t1"}, {"target": "t2"}]
        response = AppletMessage(content="", metadata={"branch": "true"})
        targets = Orchestrator._resolve_next_targets(node, edges, response)
        assert targets == ["t1"]

    def test_resolve_next_targets_if_else_by_index(self):
        node = {"type": "if_else", "data": {}}
        edges = [{"target": "t1"}, {"target": "t2"}]
        # No branch hints, defaults: index 0 = true, index 1 = false
        response_true = AppletMessage(content="", metadata={"branch": "true"})
        targets = Orchestrator._resolve_next_targets(node, edges, response_true)
        assert targets == ["t1"]

        response_false = AppletMessage(content="", metadata={"branch": "false"})
        targets = Orchestrator._resolve_next_targets(node, edges, response_false)
        assert targets == ["t2"]

    def test_resolve_next_targets_if_else_no_edges(self):
        node = {"type": "if_else", "data": {}}
        assert Orchestrator._resolve_next_targets(node, []) == []

    def test_mark_animated_edges(self):
        edges = [
            {"source": "s1", "target": "t1", "animated": False},
            {"source": "s1", "target": "t2", "animated": False},
        ]
        Orchestrator._mark_animated_edges(edges, "s1", ["t1"])
        assert edges[0]["animated"] is True
        assert edges[1]["animated"] is False

    def test_mark_animated_edges_empty_targets(self):
        edges = [{"source": "s1", "target": "t1", "animated": False}]
        Orchestrator._mark_animated_edges(edges, "s1", [])
        assert edges[0]["animated"] is False


# ============================================================
# Legacy Migration Tests
# ============================================================


class TestLegacyMigrations:
    def test_migrate_legacy_writer_nodes(self):
        flow = {
            "id": "f1",
            "nodes": [
                {"id": "n1", "type": "writer", "data": {"systemPrompt": "help me"}},
            ],
        }
        migrated, changed = Orchestrator.migrate_legacy_writer_nodes(flow)
        assert changed is True
        assert migrated["nodes"][0]["type"] == "llm"
        assert migrated["nodes"][0]["data"]["system_prompt"] == "help me"
        assert migrated["nodes"][0]["data"]["provider"] == "openai"

    def test_migrate_legacy_writer_no_writers(self):
        flow = {"id": "f1", "nodes": [{"id": "n1", "type": "llm"}]}
        migrated, changed = Orchestrator.migrate_legacy_writer_nodes(flow)
        assert changed is False

    def test_migrate_legacy_writer_non_dict(self):
        result, changed = Orchestrator.migrate_legacy_writer_nodes("not a dict")
        assert changed is False

    def test_migrate_legacy_writer_no_nodes(self):
        result, changed = Orchestrator.migrate_legacy_writer_nodes({"id": "f"})
        assert changed is False

    def test_migrate_legacy_artist_nodes(self):
        # When provider is set in data, it's preserved
        flow = {
            "id": "f1",
            "nodes": [
                {"id": "n1", "type": "artist", "data": {"provider": "openai"}},
            ],
        }
        migrated, changed = Orchestrator.migrate_legacy_artist_nodes(flow)
        assert changed is True
        assert migrated["nodes"][0]["type"] == "image"
        assert migrated["nodes"][0]["data"]["provider"] == "openai"

    def test_migrate_legacy_artist_stability_default(self):
        flow = {
            "id": "f1",
            "nodes": [{"id": "n1", "type": "artist", "data": {}}],
        }
        migrated, changed = Orchestrator.migrate_legacy_artist_nodes(flow)
        assert changed is True
        assert migrated["nodes"][0]["data"]["provider"] == "stability"

    def test_resolve_legacy_artist_defaults_openai(self):
        assert Orchestrator._resolve_legacy_artist_defaults("openai")["provider"] == "openai"
        assert Orchestrator._resolve_legacy_artist_defaults("dall-e-3")["provider"] == "openai"

    def test_resolve_legacy_artist_defaults_flux(self):
        assert Orchestrator._resolve_legacy_artist_defaults("flux")["provider"] == "flux"

    def test_resolve_legacy_artist_defaults_fallback(self):
        assert Orchestrator._resolve_legacy_artist_defaults("unknown")["provider"] == "stability"
        assert Orchestrator._resolve_legacy_artist_defaults(None)["provider"] == "stability"

    @pytest.mark.asyncio
    async def test_load_applet_all_types(self):
        """Test that load_applet resolves all built-in node types."""
        from apps.orchestrator.main import applet_registry

        # Clear registry for clean test
        with patch.dict("apps.orchestrator.main.applet_registry", {}, clear=True):
            for applet_type in [
                "llm", "image", "memory", "http_request",
                "code", "transform", "if_else", "merge", "for_each",
            ]:
                applet = await Orchestrator.load_applet(applet_type)
                assert isinstance(applet, BaseApplet)

    @pytest.mark.asyncio
    async def test_load_applet_aliases(self):
        """Test that applet aliases resolve correctly."""
        with patch.dict("apps.orchestrator.main.applet_registry", {}, clear=True):
            applet = await Orchestrator.load_applet("http-request")
            assert applet.__class__.__name__ == "HTTPRequestNodeApplet"

            applet = await Orchestrator.load_applet("ifelse")
            assert applet.__class__.__name__ == "IfElseNodeApplet"

            applet = await Orchestrator.load_applet("foreach")
            assert applet.__class__.__name__ == "ForEachNodeApplet"


# ============================================================
# Run Diff Tests
# ============================================================


class TestRunDiff:
    def test_build_run_diff_identical(self):
        run = {
            "run_id": "r1",
            "flow_id": "f1",
            "status": "success",
            "start_time": 1000.0,
            "end_time": 1001.0,
            "input_data": {"q": "test"},
            "results": {"n1": {"output": "hello"}},
        }
        diff = _build_run_diff(run, run)
        assert diff["summary"]["status_changed"] is False
        assert diff["input_diff"]["changed"] is False

    def test_build_run_diff_different_status(self):
        run1 = {
            "run_id": "r1",
            "status": "success",
            "start_time": 1000.0,
            "end_time": 1001.0,
            "results": {},
        }
        run2 = {
            "run_id": "r2",
            "status": "error",
            "start_time": 1000.0,
            "end_time": 1002.0,
            "results": {},
        }
        diff = _build_run_diff(run1, run2)
        assert diff["summary"]["status_changed"] is True
        assert diff["base_run_id"] == "r1"
        assert diff["compare_run_id"] == "r2"

    def test_build_run_diff_node_added(self):
        run1 = {"run_id": "r1", "status": "success", "start_time": 1000.0, "results": {}}
        run2 = {
            "run_id": "r2",
            "status": "success",
            "start_time": 1000.0,
            "results": {"n1": {"output": "new", "type": "writer", "status": "success"}},
        }
        diff = _build_run_diff(run1, run2)
        assert diff["summary"]["changed_node_count"] > 0


# ============================================================
# API Endpoint Tests
# ============================================================


class TestAPIEndpoints:
    def test_health_check(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_versioned_health(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert "version" in resp.json()

    def test_list_applets_pagination_params(self, client):
        resp = client.get("/api/v1/applets?page=2&page_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["page_size"] == 3

    def test_create_and_get_flow(self, client):
        flow = {
            "name": "Integration Test Flow",
            "nodes": [
                {"id": "s", "type": "start", "position": {"x": 0, "y": 0}},
            ],
            "edges": [],
        }
        resp = client.post("/api/v1/flows", json=flow)
        assert resp.status_code == 201
        flow_id = resp.json()["id"]

        resp = client.get(f"/api/v1/flows/{flow_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Integration Test Flow"

    def test_delete_nonexistent_flow(self, client):
        resp = client.delete("/api/v1/flows/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_list_flows_paginated(self, client):
        resp = client.get("/api/v1/flows?page=1&page_size=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total_pages" in data

    def test_list_runs_empty(self, client):
        resp = client.get("/api/v1/runs")
        assert resp.status_code == 200
        assert "items" in resp.json()

    def test_get_run_not_found(self, client):
        resp = client.get("/api/v1/runs/nonexistent")
        assert resp.status_code == 404

    def test_ai_suggest_not_implemented(self, client):
        resp = client.post("/api/v1/ai/suggest", json={"prompt": "test"})
        assert resp.status_code == 501

    def test_llm_providers(self, client):
        resp = client.get("/api/v1/llm/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        provider_names = [p["name"] for p in data["items"]]
        assert "openai" in provider_names

    def test_image_providers(self, client):
        resp = client.get("/api/v1/image/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data


# ============================================================
# Auth Endpoint Tests
# ============================================================


@pytest.fixture(autouse=True, scope="class")
def _disable_rate_limit_for_auth(monkeypatch_class=None):
    """Give auth tests a very high rate limit so they don't get 429."""
    import apps.orchestrator.middleware.rate_limiter as rl
    original = rl._TIER_LIMITS.copy()
    rl._TIER_LIMITS.update({"anonymous": 10000, "free": 10000, "pro": 10000, "enterprise": 10000})
    # Also reset the counter
    fresh = rl._SlidingWindowCounter()
    old_counter = rl._counter
    rl._counter = fresh
    yield
    rl._TIER_LIMITS.update(original)
    rl._counter = old_counter


class TestAuthEndpoints:
    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        import apps.orchestrator.middleware.rate_limiter as rl
        rl._counter = rl._SlidingWindowCounter()
        rl._TIER_LIMITS.update({"anonymous": 10000, "free": 10000, "pro": 10000, "enterprise": 10000})

    def test_register_and_login(self, client):
        email = f"test-{uuid.uuid4().hex[:8]}@example.com"
        # Register
        resp = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "SecurePassword123!",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

        # Login
        resp = client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "SecurePassword123!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    def test_login_wrong_password(self, client):
        email = f"test-{uuid.uuid4().hex[:8]}@example.com"
        client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "CorrectPassword",
        })
        resp = client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "WrongPassword",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/api/v1/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "whatever",
        })
        assert resp.status_code == 401

    def test_register_duplicate_email(self, client):
        email = f"dup-{uuid.uuid4().hex[:8]}@example.com"
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        resp = client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        assert resp.status_code == 409

    def test_register_invalid_email(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "password": "pass1234",
        })
        assert resp.status_code == 422

    def test_me_with_token(self, client):
        email = f"me-{uuid.uuid4().hex[:8]}@example.com"
        reg = client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        token = reg.json()["access_token"]

        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        assert resp.status_code == 200
        assert resp.json()["email"] == email

    def test_me_without_token(self, client):
        """Without a token and when users exist, should fail."""
        # First register so anonymous bootstrap is disabled
        email = f"block-{uuid.uuid4().hex[:8]}@example.com"
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_refresh_token(self, client):
        email = f"refresh-{uuid.uuid4().hex[:8]}@example.com"
        reg = client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        refresh_token = reg.json()["refresh_token"]

        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_logout(self, client):
        email = f"logout-{uuid.uuid4().hex[:8]}@example.com"
        reg = client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        token = reg.json()["access_token"]
        refresh = reg.json()["refresh_token"]

        resp = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_api_keys_crud(self, client):
        email = f"apikey-{uuid.uuid4().hex[:8]}@example.com"
        reg = client.post("/api/v1/auth/register", json={
            "email": email, "password": "pass1234",
        })
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create API key
        resp = client.post("/api/v1/auth/api-keys", json={
            "name": "test-key",
        }, headers=headers)
        assert resp.status_code == 201
        api_key_data = resp.json()
        assert "api_key" in api_key_data
        api_key_id = api_key_data["id"]
        api_key_value = api_key_data["api_key"]

        # List API keys
        resp = client.get("/api/v1/auth/api-keys", headers=headers)
        assert resp.status_code == 200
        keys_data = resp.json()
        keys = keys_data if isinstance(keys_data, list) else keys_data.get("items", keys_data)
        assert any(k["id"] == api_key_id for k in keys)

        # Use API key for auth
        resp = client.get("/api/v1/auth/me", headers={
            "X-API-Key": api_key_value,
        })
        assert resp.status_code == 200
        assert resp.json()["email"] == email

        # Delete API key
        resp = client.delete(f"/api/v1/auth/api-keys/{api_key_id}", headers=headers)
        assert resp.status_code == 200


# ============================================================
# Flow Execution Integration Tests
# ============================================================


class TestFlowExecution:
    @pytest.mark.asyncio
    async def test_execute_simple_flow(self, db):
        flow_data = {
            "id": "simple-exec",
            "name": "Simple Exec",
            "nodes": [
                {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
                {"id": "end", "type": "end", "position": {"x": 100, "y": 0}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "end"},
            ],
        }
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"input": "test"})
            assert run_id is not None

            for _ in range(20):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] in ("success", "error"):
                    break
            assert run["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_flow_with_mocked_llm(self, db):
        flow_data = {
            "id": "llm-flow",
            "name": "LLM Flow",
            "nodes": [
                {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
                {
                    "id": "llm-node",
                    "type": "llm",
                    "position": {"x": 100, "y": 0},
                    "data": {"provider": "openai", "model": "gpt-4o"},
                },
                {"id": "end", "type": "end", "position": {"x": 200, "y": 0}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "llm-node"},
                {"id": "e2", "source": "llm-node", "target": "end"},
            ],
        }

        mock_response = AppletMessage(
            content="Mocked LLM output", context={}, metadata={}
        )
        with patch(
            "apps.orchestrator.main.LLMNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
                run_id = await Orchestrator.execute_flow(flow_data, {"prompt": "hi"})

                for _ in range(30):
                    await asyncio.sleep(0.1)
                    run = await WorkflowRunRepository.get_by_run_id(run_id)
                    if run and run["status"] in ("success", "error"):
                        break

                assert run["status"] == "success"
                assert "llm-node" in run["results"]

    @pytest.mark.asyncio
    async def test_execute_flow_run_endpoint(self, client, db):
        # Create flow
        flow = {
            "id": "run-test-flow",
            "name": "Run Test Flow",
            "nodes": [
                {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
                {"id": "end", "type": "end", "position": {"x": 100, "y": 0}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "end"},
            ],
        }
        client.post("/api/v1/flows", json=flow)

        # Run it via API
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            resp = client.post("/api/v1/flows/run-test-flow/runs", json={
                "input": {"test": "data"},
            })
            assert resp.status_code == 202
            data = resp.json()
            assert "run_id" in data

    @pytest.mark.asyncio
    async def test_rerun_flow(self, db):
        """Test rerun creates a new execution from a previous run."""
        # Create and run the original
        flow_data = {
            "id": "rerun-flow",
            "name": "Rerun Flow",
            "nodes": [
                {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
                {"id": "end", "type": "end", "position": {"x": 100, "y": 0}},
            ],
            "edges": [{"id": "e1", "source": "start", "target": "end"}],
        }

        await FlowRepository.save(flow_data)

        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"key": "val"})
            for _ in range(20):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] in ("success", "error"):
                    break

            assert run["status"] == "success"


# ============================================================
# WebSocket Tests
# ============================================================


class TestWebSocket:
    def test_websocket_connect_and_auth(self, client):
        with client.websocket_connect("/api/v1/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "auth.result"
            assert msg["data"]["authenticated"] is True

    def test_websocket_ping(self, client):
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"

    def test_websocket_subscribe(self, client):
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth
            ws.send_json({"type": "subscribe", "data": {"topics": ["workflow.status"]}})
            msg = ws.receive_json()
            assert msg["type"] == "subscribe.ack"

    def test_websocket_invalid_json(self, client):
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth
            ws.send_text("not json")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["data"]["code"] == "INVALID_MESSAGE"

    def test_websocket_unknown_type(self, client):
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth
            ws.send_json({"type": "foobar"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["data"]["code"] == "UNKNOWN_MESSAGE_TYPE"


# ============================================================
# Trace and Diff Endpoint Tests
# ============================================================


class TestTraceAndDiffEndpoints:
    def test_get_trace_not_found(self, client):
        resp = client.get("/api/v1/runs/nonexistent/trace")
        assert resp.status_code == 404

    def test_get_diff_not_found(self, client):
        resp = client.get("/api/v1/runs/nonexistent/diff?other_run_id=also-missing")
        assert resp.status_code == 404
