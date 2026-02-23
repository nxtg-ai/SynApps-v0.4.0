"""
Tests for DIRECTIVE-NXTG-20260223-07 — Error Classification + Retry Policies.

Covers:
- ErrorCategory enum values
- classify_error() with status codes and exceptions
- RetryPolicy: should_retry, delay_for_attempt, serialization
- Per-connector policies and get_retry_policy()
- ConnectorError exception fields
- execute_with_retry(): transient retry, permanent fail-fast, rate-limited backoff,
  exhaustion, success after retries
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from apps.orchestrator.main import (
    ErrorCategory,
    RetryPolicy,
    ConnectorError,
    classify_error,
    get_retry_policy,
    execute_with_retry,
    CONNECTOR_RETRY_POLICIES,
    DEFAULT_RETRY_POLICY,
)


# ---------------------------------------------------------------------------
# ErrorCategory enum
# ---------------------------------------------------------------------------

class TestErrorCategory:

    def test_values(self):
        assert ErrorCategory.TRANSIENT == "transient"
        assert ErrorCategory.RATE_LIMITED == "rate_limited"
        assert ErrorCategory.PERMANENT == "permanent"

    def test_is_string_enum(self):
        assert isinstance(ErrorCategory.TRANSIENT, str)


# ---------------------------------------------------------------------------
# classify_error()
# ---------------------------------------------------------------------------

class TestClassifyError:

    # Status code classification
    def test_429_is_rate_limited(self):
        assert classify_error(status_code=429) == ErrorCategory.RATE_LIMITED

    def test_500_is_transient(self):
        assert classify_error(status_code=500) == ErrorCategory.TRANSIENT

    def test_502_is_transient(self):
        assert classify_error(status_code=502) == ErrorCategory.TRANSIENT

    def test_503_is_transient(self):
        assert classify_error(status_code=503) == ErrorCategory.TRANSIENT

    def test_504_is_transient(self):
        assert classify_error(status_code=504) == ErrorCategory.TRANSIENT

    def test_408_is_transient(self):
        assert classify_error(status_code=408) == ErrorCategory.TRANSIENT

    def test_401_is_permanent(self):
        assert classify_error(status_code=401) == ErrorCategory.PERMANENT

    def test_403_is_permanent(self):
        assert classify_error(status_code=403) == ErrorCategory.PERMANENT

    def test_404_is_permanent(self):
        assert classify_error(status_code=404) == ErrorCategory.PERMANENT

    def test_422_is_permanent(self):
        assert classify_error(status_code=422) == ErrorCategory.PERMANENT

    def test_unknown_status_is_permanent(self):
        assert classify_error(status_code=418) == ErrorCategory.PERMANENT

    # Exception classification
    def test_connect_error_is_transient(self):
        assert classify_error(exc=httpx.ConnectError("refused")) == ErrorCategory.TRANSIENT

    def test_timeout_error_is_transient(self):
        assert classify_error(exc=TimeoutError("timed out")) == ErrorCategory.TRANSIENT

    def test_connection_error_is_transient(self):
        assert classify_error(exc=ConnectionError("reset")) == ErrorCategory.TRANSIENT

    def test_os_error_is_transient(self):
        assert classify_error(exc=OSError("broken pipe")) == ErrorCategory.TRANSIENT

    def test_read_timeout_is_transient(self):
        assert classify_error(exc=httpx.ReadTimeout("read")) == ErrorCategory.TRANSIENT

    def test_value_error_is_permanent(self):
        assert classify_error(exc=ValueError("bad input")) == ErrorCategory.PERMANENT

    def test_runtime_error_is_permanent(self):
        assert classify_error(exc=RuntimeError("crash")) == ErrorCategory.PERMANENT

    # httpx.HTTPStatusError
    def test_http_status_error_429(self):
        resp = httpx.Response(429, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("rate limit", request=resp.request, response=resp)
        assert classify_error(exc=exc) == ErrorCategory.RATE_LIMITED

    def test_http_status_error_500(self):
        resp = httpx.Response(500, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("server err", request=resp.request, response=resp)
        assert classify_error(exc=exc) == ErrorCategory.TRANSIENT

    # Status code takes priority over exception
    def test_status_code_priority_over_exception(self):
        """If both status_code and exc are given, status_code wins."""
        assert classify_error(exc=TimeoutError(), status_code=404) == ErrorCategory.PERMANENT

    # No info → permanent
    def test_no_info_is_permanent(self):
        assert classify_error() == ErrorCategory.PERMANENT


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class TestRetryPolicy:

    def test_default_retries(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 1.0
        assert p.backoff_factor == 2.0

    def test_should_retry_transient(self):
        p = RetryPolicy(max_retries=3)
        assert p.should_retry(ErrorCategory.TRANSIENT, attempt=0) is True
        assert p.should_retry(ErrorCategory.TRANSIENT, attempt=2) is True
        assert p.should_retry(ErrorCategory.TRANSIENT, attempt=3) is False  # exhausted

    def test_should_retry_rate_limited(self):
        p = RetryPolicy(max_retries=3)
        assert p.should_retry(ErrorCategory.RATE_LIMITED, attempt=0) is True

    def test_should_not_retry_permanent(self):
        p = RetryPolicy(max_retries=3)
        assert p.should_retry(ErrorCategory.PERMANENT, attempt=0) is False

    def test_delay_for_attempt_exponential(self):
        p = RetryPolicy(base_delay=1.0, backoff_factor=2.0)
        assert p.delay_for_attempt(0) == 1.0
        assert p.delay_for_attempt(1) == 2.0
        assert p.delay_for_attempt(2) == 4.0
        assert p.delay_for_attempt(3) == 8.0

    def test_custom_base_delay(self):
        p = RetryPolicy(base_delay=0.5, backoff_factor=3.0)
        assert p.delay_for_attempt(0) == 0.5
        assert p.delay_for_attempt(1) == 1.5

    def test_to_dict(self):
        p = RetryPolicy(max_retries=2, base_delay=0.5, backoff_factor=3.0)
        d = p.to_dict()
        assert d["max_retries"] == 2
        assert d["base_delay"] == 0.5
        assert d["backoff_factor"] == 3.0
        assert "transient" in d["retryable_categories"]
        assert "rate_limited" in d["retryable_categories"]

    def test_custom_retryable_categories(self):
        p = RetryPolicy(retryable_categories={ErrorCategory.TRANSIENT})
        assert p.should_retry(ErrorCategory.TRANSIENT, attempt=0) is True
        assert p.should_retry(ErrorCategory.RATE_LIMITED, attempt=0) is False

    def test_zero_retries(self):
        p = RetryPolicy(max_retries=0)
        assert p.should_retry(ErrorCategory.TRANSIENT, attempt=0) is False


# ---------------------------------------------------------------------------
# Per-connector policies
# ---------------------------------------------------------------------------

class TestConnectorPolicies:

    def test_known_connectors_have_policies(self):
        for name in ("openai", "anthropic", "google", "ollama", "custom", "stability"):
            policy = get_retry_policy(name)
            assert isinstance(policy, RetryPolicy)
            assert policy.max_retries >= 1

    def test_unknown_connector_gets_default(self):
        policy = get_retry_policy("unknown-provider")
        assert policy is DEFAULT_RETRY_POLICY

    def test_ollama_fewer_retries(self):
        """Ollama is local — fewer retries expected."""
        assert CONNECTOR_RETRY_POLICIES["ollama"].max_retries <= 2

    def test_stability_higher_base_delay(self):
        """Image gen is slow — higher base delay expected."""
        assert CONNECTOR_RETRY_POLICIES["stability"].base_delay >= 2.0


# ---------------------------------------------------------------------------
# ConnectorError
# ---------------------------------------------------------------------------

class TestConnectorError:

    def test_fields(self):
        err = ConnectorError(
            "rate limited",
            category=ErrorCategory.RATE_LIMITED,
            connector="openai",
            status_code=429,
            attempt=2,
            max_retries=3,
        )
        assert err.category == ErrorCategory.RATE_LIMITED
        assert err.connector == "openai"
        assert err.status_code == 429
        assert err.attempt == 2
        assert err.max_retries == 3
        assert "rate limited" in str(err)

    def test_inherits_exception(self):
        err = ConnectorError("x", category=ErrorCategory.PERMANENT)
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# execute_with_retry()
# ---------------------------------------------------------------------------

class TestExecuteWithRetry:

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        func = AsyncMock(return_value="ok")
        result = await execute_with_retry(func, connector="openai")
        assert result == "ok"
        assert func.await_count == 1

    @pytest.mark.asyncio
    async def test_success_after_transient_failures(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("reset")
            return "recovered"

        policy = RetryPolicy(max_retries=3, base_delay=0.01, backoff_factor=1.0)
        result = await execute_with_retry(flaky, connector="test", policy=policy)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_permanent_error_fails_immediately(self):
        async def auth_fail():
            raise ValueError("invalid API key")

        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        with pytest.raises(ConnectorError) as exc_info:
            await execute_with_retry(auth_fail, connector="openai", policy=policy)
        err = exc_info.value
        assert err.category == ErrorCategory.PERMANENT
        assert err.attempt == 0  # failed on first attempt

    @pytest.mark.asyncio
    async def test_rate_limited_retries_with_backoff(self):
        attempts = []

        async def rate_limited():
            attempts.append(time.monotonic())
            resp = httpx.Response(429, request=httpx.Request("GET", "http://x"))
            raise httpx.HTTPStatusError("rate limit", request=resp.request, response=resp)

        policy = RetryPolicy(max_retries=2, base_delay=0.05, backoff_factor=2.0)
        with pytest.raises(ConnectorError) as exc_info:
            await execute_with_retry(rate_limited, connector="openai", policy=policy)
        err = exc_info.value
        assert err.category == ErrorCategory.RATE_LIMITED
        assert len(attempts) == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_exhaustion_raises_connector_error(self):
        async def always_fail():
            raise ConnectionError("down")

        policy = RetryPolicy(max_retries=2, base_delay=0.01, backoff_factor=1.0)
        with pytest.raises(ConnectorError) as exc_info:
            await execute_with_retry(always_fail, connector="test", policy=policy)
        err = exc_info.value
        assert err.category == ErrorCategory.TRANSIENT
        assert err.connector == "test"

    @pytest.mark.asyncio
    async def test_zero_retries_policy(self):
        async def fail():
            raise ConnectionError("down")

        policy = RetryPolicy(max_retries=0, base_delay=0.01)
        with pytest.raises(ConnectorError) as exc_info:
            await execute_with_retry(fail, connector="test", policy=policy)
        assert exc_info.value.attempt == 0

    @pytest.mark.asyncio
    async def test_http_status_error_classification(self):
        """HTTPStatusError with 503 should be retried as transient."""
        call_count = 0

        async def server_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                resp = httpx.Response(503, request=httpx.Request("GET", "http://x"))
                raise httpx.HTTPStatusError("unavailable", request=resp.request, response=resp)
            return "back up"

        policy = RetryPolicy(max_retries=3, base_delay=0.01, backoff_factor=1.0)
        result = await execute_with_retry(server_error, connector="test", policy=policy)
        assert result == "back up"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_uses_connector_default_policy(self):
        """When no policy is given, uses per-connector default."""
        async def ok():
            return "done"

        result = await execute_with_retry(ok, connector="anthropic")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_connector_error_preserves_cause(self):
        original = ConnectionError("root cause")

        async def fail():
            raise original

        policy = RetryPolicy(max_retries=0, base_delay=0.01)
        with pytest.raises(ConnectorError) as exc_info:
            await execute_with_retry(fail, connector="x", policy=policy)
        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_status_code_on_connector_error(self):
        async def fail():
            resp = httpx.Response(429, request=httpx.Request("GET", "http://x"))
            raise httpx.HTTPStatusError("rate", request=resp.request, response=resp)

        policy = RetryPolicy(max_retries=0, base_delay=0.01)
        with pytest.raises(ConnectorError) as exc_info:
            await execute_with_retry(fail, connector="openai", policy=policy)
        assert exc_info.value.status_code == 429
