"""
Tests for DIRECTIVE-NXTG-20260223-08 — Connector Health Probes.

Covers:
- ConnectorStatus enum
- ConnectorHealthTracker: record_success, record_failure, auto-disable,
  auto-re-enable, get_status, is_disabled, all_statuses, reset
- probe_connector(): success, failure, unknown connector
- probe_all_connectors(): returns all known connectors
- GET /connectors/health endpoint: structure, summary counts
- POST /connectors/{name}/probe endpoint
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    ConnectorHealthTracker,
    ConnectorStatus,
    connector_health,
    probe_connector,
    probe_all_connectors,
    app,
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Reset global connector_health between tests."""
    connector_health.reset()
    yield
    connector_health.reset()


# ---------------------------------------------------------------------------
# ConnectorStatus enum
# ---------------------------------------------------------------------------

class TestConnectorStatus:

    def test_values(self):
        assert ConnectorStatus.HEALTHY == "healthy"
        assert ConnectorStatus.DEGRADED == "degraded"
        assert ConnectorStatus.DISABLED == "disabled"


# ---------------------------------------------------------------------------
# ConnectorHealthTracker unit tests
# ---------------------------------------------------------------------------

class TestConnectorHealthTracker:

    def test_new_connector_is_healthy(self):
        t = ConnectorHealthTracker()
        s = t.get_status("test")
        assert s["status"] == ConnectorStatus.HEALTHY
        assert s["consecutive_failures"] == 0

    def test_record_success(self):
        t = ConnectorHealthTracker()
        t.record_success("openai")
        s = t.get_status("openai")
        assert s["status"] == ConnectorStatus.HEALTHY
        assert s["total_probes"] == 1
        assert s["last_check"] is not None

    def test_single_failure_is_degraded(self):
        t = ConnectorHealthTracker()
        t.record_failure("openai", "timeout")
        s = t.get_status("openai")
        assert s["status"] == ConnectorStatus.DEGRADED
        assert s["consecutive_failures"] == 1
        assert s["last_failure_reason"] == "timeout"

    def test_two_failures_still_degraded(self):
        t = ConnectorHealthTracker(disable_threshold=3)
        t.record_failure("openai", "fail1")
        t.record_failure("openai", "fail2")
        s = t.get_status("openai")
        assert s["status"] == ConnectorStatus.DEGRADED
        assert s["consecutive_failures"] == 2

    def test_three_failures_is_disabled(self):
        t = ConnectorHealthTracker(disable_threshold=3)
        for i in range(3):
            t.record_failure("openai", f"fail{i}")
        s = t.get_status("openai")
        assert s["status"] == ConnectorStatus.DISABLED
        assert s["consecutive_failures"] == 3

    def test_more_failures_stays_disabled(self):
        t = ConnectorHealthTracker(disable_threshold=3)
        for i in range(5):
            t.record_failure("openai", f"fail{i}")
        s = t.get_status("openai")
        assert s["status"] == ConnectorStatus.DISABLED
        assert s["consecutive_failures"] == 5

    def test_success_after_disable_re_enables(self):
        t = ConnectorHealthTracker(disable_threshold=3)
        for i in range(3):
            t.record_failure("openai", f"fail{i}")
        assert t.is_disabled("openai")
        t.record_success("openai")
        s = t.get_status("openai")
        assert s["status"] == ConnectorStatus.HEALTHY
        assert s["consecutive_failures"] == 0
        assert not t.is_disabled("openai")

    def test_success_clears_failure_reason(self):
        t = ConnectorHealthTracker()
        t.record_failure("openai", "some reason")
        assert t.get_status("openai")["last_failure_reason"] == "some reason"
        t.record_success("openai")
        assert t.get_status("openai")["last_failure_reason"] is None

    def test_is_disabled(self):
        t = ConnectorHealthTracker(disable_threshold=2)
        assert t.is_disabled("x") is False
        t.record_failure("x", "a")
        assert t.is_disabled("x") is False
        t.record_failure("x", "b")
        assert t.is_disabled("x") is True

    def test_all_statuses(self):
        t = ConnectorHealthTracker()
        t.record_success("openai")
        t.record_failure("anthropic", "err")
        statuses = t.all_statuses()
        assert "openai" in statuses
        assert "anthropic" in statuses
        assert statuses["openai"]["status"] == ConnectorStatus.HEALTHY
        assert statuses["anthropic"]["status"] == ConnectorStatus.DEGRADED

    def test_all_statuses_empty(self):
        t = ConnectorHealthTracker()
        assert t.all_statuses() == {}

    def test_total_failures_cumulative(self):
        t = ConnectorHealthTracker()
        t.record_failure("x", "a")
        t.record_success("x")
        t.record_failure("x", "b")
        s = t.get_status("x")
        assert s["total_failures"] == 2
        assert s["total_probes"] == 3
        assert s["consecutive_failures"] == 1  # reset by success

    def test_reset(self):
        t = ConnectorHealthTracker()
        t.record_success("openai")
        t.record_failure("anthropic", "err")
        t.reset()
        assert t.all_statuses() == {}

    def test_disable_threshold_property(self):
        t = ConnectorHealthTracker(disable_threshold=5)
        assert t.disable_threshold == 5

    def test_custom_threshold(self):
        t = ConnectorHealthTracker(disable_threshold=1)
        t.record_failure("x", "one strike")
        assert t.is_disabled("x")


# ---------------------------------------------------------------------------
# probe_connector() tests
# ---------------------------------------------------------------------------

class TestProbeConnector:

    @pytest.mark.asyncio
    async def test_probe_configured_provider(self):
        """Probing a provider with a valid API key should return reachable=True."""
        # openai adapter's validate_config checks OPENAI_API_KEY
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from apps.orchestrator.main import LLMProviderRegistry
            result = await probe_connector("openai")
            assert result["connector"] == "openai"
            assert result["reachable"] is True

    @pytest.mark.asyncio
    async def test_probe_unconfigured_provider(self):
        """Probing a provider without API key should return reachable=False."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove the key if present
            import os
            old = os.environ.pop("OPENAI_API_KEY", None)
            try:
                result = await probe_connector("openai")
                assert result["connector"] == "openai"
                assert result["reachable"] is False
                assert "not set" in result["detail"].lower() or result["detail"] != ""
            finally:
                if old:
                    os.environ["OPENAI_API_KEY"] = old

    @pytest.mark.asyncio
    async def test_probe_unknown_connector(self):
        result = await probe_connector("nonexistent-provider")
        assert result["connector"] == "nonexistent-provider"
        assert result["reachable"] is False
        assert "unknown" in result["detail"].lower()

    @pytest.mark.asyncio
    async def test_probe_records_in_tracker(self):
        connector_health.reset()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await probe_connector("openai")
        s = connector_health.get_status("openai")
        assert s["total_probes"] >= 1


# ---------------------------------------------------------------------------
# probe_all_connectors() tests
# ---------------------------------------------------------------------------

class TestProbeAllConnectors:

    @pytest.mark.asyncio
    async def test_returns_all_known_connectors(self):
        connector_health.reset()
        results = await probe_all_connectors()
        names = {r["connector"] for r in results}
        assert "openai" in names
        assert "anthropic" in names
        assert "google" in names
        assert "ollama" in names

    @pytest.mark.asyncio
    async def test_each_result_has_required_fields(self):
        connector_health.reset()
        results = await probe_all_connectors()
        for r in results:
            assert "connector" in r
            assert "reachable" in r
            assert "status" in r
            assert "consecutive_failures" in r
            assert "total_probes" in r


# ---------------------------------------------------------------------------
# GET /connectors/health endpoint
# ---------------------------------------------------------------------------

class TestConnectorsHealthEndpoint:

    def test_returns_200(self, client):
        resp = client.get("/api/v1/connectors/health")
        assert resp.status_code == 200

    def test_has_connectors_list(self, client):
        data = client.get("/api/v1/connectors/health").json()
        assert "connectors" in data
        assert isinstance(data["connectors"], list)
        assert len(data["connectors"]) > 0

    def test_has_summary(self, client):
        data = client.get("/api/v1/connectors/health").json()
        summary = data["summary"]
        assert "healthy" in summary
        assert "degraded" in summary
        assert "disabled" in summary
        assert summary["healthy"] + summary["degraded"] + summary["disabled"] == data["total"]

    def test_has_disable_threshold(self, client):
        data = client.get("/api/v1/connectors/health").json()
        assert data["disable_threshold"] == 3

    def test_connector_result_structure(self, client):
        data = client.get("/api/v1/connectors/health").json()
        c = data["connectors"][0]
        assert "connector" in c
        assert "reachable" in c
        assert "status" in c
        assert "consecutive_failures" in c
        assert "total_probes" in c
        assert "total_failures" in c
        assert "last_check" in c

    def test_known_connectors_present(self, client):
        data = client.get("/api/v1/connectors/health").json()
        names = {c["connector"] for c in data["connectors"]}
        assert "openai" in names


# ---------------------------------------------------------------------------
# POST /connectors/{name}/probe endpoint
# ---------------------------------------------------------------------------

class TestProbeEndpoint:

    def test_probe_known_connector(self, client):
        resp = client.post("/api/v1/connectors/openai/probe")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connector"] == "openai"
        assert "status" in data
        assert "consecutive_failures" in data

    def test_probe_unknown_connector(self, client):
        resp = client.post("/api/v1/connectors/nonexistent/probe")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connector"] == "nonexistent"
        assert data["reachable"] is False


# ---------------------------------------------------------------------------
# Auto-disable / auto-re-enable integration
# ---------------------------------------------------------------------------

class TestAutoDisableIntegration:

    @pytest.mark.asyncio
    async def test_repeated_failures_disable_connector(self):
        connector_health.reset()
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            for _ in range(3):
                await probe_connector("openai")
            assert connector_health.is_disabled("openai")
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old

    @pytest.mark.asyncio
    async def test_success_re_enables_disabled_connector(self):
        connector_health.reset()
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Disable with 3 failures
            for _ in range(3):
                await probe_connector("openai")
            assert connector_health.is_disabled("openai")

            # Re-enable with valid key
            os.environ["OPENAI_API_KEY"] = "sk-test"
            await probe_connector("openai")
            assert not connector_health.is_disabled("openai")
            assert connector_health.get_status("openai")["status"] == ConnectorStatus.HEALTHY
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old
            else:
                os.environ.pop("OPENAI_API_KEY", None)
