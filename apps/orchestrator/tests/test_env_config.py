"""Tests for environment configuration, /config endpoint, and startup validation (DIRECTIVE-23-02)."""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, AppConfig, _redact, _SECRET_KEYS, app_config


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# _redact helper
# ---------------------------------------------------------------------------


def test_redact_short_value():
    """Short secrets are fully redacted."""
    assert _redact("abc") == "***"
    assert _redact("") == "***"


def test_redact_long_value():
    """Long secrets show first 4 and last 2 chars."""
    assert _redact("sk-1234567890abcdef") == "sk-1***ef"


def test_redact_exactly_8():
    """8-char value is redacted with prefix/suffix."""
    assert _redact("12345678") == "1234***78"


# ---------------------------------------------------------------------------
# AppConfig — unit tests
# ---------------------------------------------------------------------------


def test_app_config_defaults():
    """AppConfig loads defaults when no env vars are set."""
    with patch.dict(os.environ, {}, clear=False):
        config = AppConfig()
    assert config.backend_port == 8000
    assert config.production is False
    assert config.debug is False
    assert config.log_level == "info"
    assert config.rate_limit_free == 60
    assert config.engine_max_concurrency == 10


def test_app_config_reads_env():
    """AppConfig reads values from environment."""
    env = {
        "BACKEND_PORT": "9000",
        "PRODUCTION": "true",
        "DEBUG": "true",
        "LOG_LEVEL": "debug",
        "RATE_LIMIT_FREE": "100",
        "ENGINE_MAX_CONCURRENCY": "5",
    }
    with patch.dict(os.environ, env):
        config = AppConfig()
    assert config.backend_port == 9000
    assert config.production is True
    assert config.debug is True
    assert config.log_level == "debug"
    assert config.rate_limit_free == 100
    assert config.engine_max_concurrency == 5


def test_app_config_to_dict():
    """to_dict() returns all config keys."""
    config = AppConfig()
    d = config.to_dict(redact_secrets=False)
    assert "backend_port" in d
    assert "database_url" in d
    assert "jwt_secret_key" in d
    assert "rate_limit_free" in d


def test_app_config_to_dict_redacts_secrets():
    """to_dict(redact_secrets=True) redacts secret values."""
    env = {"JWT_SECRET_KEY": "my-super-secret-jwt-key-12345"}
    with patch.dict(os.environ, env):
        config = AppConfig()
    d = config.to_dict(redact_secrets=True)
    assert d["jwt_secret_key"] != "my-super-secret-jwt-key-12345"
    assert "***" in d["jwt_secret_key"]


def test_app_config_to_dict_does_not_redact_non_secrets():
    """to_dict() does not redact non-secret values."""
    config = AppConfig()
    d = config.to_dict(redact_secrets=True)
    assert isinstance(d["backend_port"], int)
    assert isinstance(d["production"], bool)
    assert isinstance(d["log_level"], str)
    assert "***" not in str(d["backend_port"])


def test_app_config_secret_keys_defined():
    """_SECRET_KEYS contains expected secret field names."""
    assert "jwt_secret_key" in _SECRET_KEYS
    assert "database_url" in _SECRET_KEYS
    assert "openai_api_key" in _SECRET_KEYS
    assert "synapps_master_key" in _SECRET_KEYS


# ---------------------------------------------------------------------------
# AppConfig.validate()
# ---------------------------------------------------------------------------


def test_validate_dev_mode_no_errors():
    """In dev mode, default config produces no errors."""
    with patch.dict(os.environ, {"PRODUCTION": "false"}, clear=False):
        config = AppConfig()
    errors = config.validate()
    assert errors == []


def test_validate_production_requires_jwt_secret():
    """In production, default JWT secret is an error."""
    env = {
        "PRODUCTION": "true",
        "JWT_SECRET_KEY": "synapps-dev-jwt-secret-change-me",
        "BACKEND_CORS_ORIGINS": "https://app.example.com",
    }
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert any("JWT_SECRET_KEY" in e for e in errors)


def test_validate_production_requires_cors():
    """In production, missing CORS origins is an error."""
    env = {
        "PRODUCTION": "true",
        "JWT_SECRET_KEY": "a-proper-secret-key-here",
        "BACKEND_CORS_ORIGINS": "",
    }
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert any("BACKEND_CORS_ORIGINS" in e for e in errors)


def test_validate_production_all_good():
    """In production with correct config, no errors."""
    env = {
        "PRODUCTION": "true",
        "JWT_SECRET_KEY": "prod-secret-key-abc123",
        "BACKEND_CORS_ORIGINS": "https://app.example.com",
    }
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert errors == []


def test_validate_invalid_port():
    """Invalid port produces an error."""
    env = {"BACKEND_PORT": "99999"}
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert any("BACKEND_PORT" in e for e in errors)


def test_validate_invalid_log_level():
    """Invalid log level produces an error."""
    env = {"LOG_LEVEL": "verbose"}
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert any("LOG_LEVEL" in e for e in errors)


def test_validate_invalid_rate_limit_window():
    """Rate limit window < 1 produces an error."""
    env = {"RATE_LIMIT_WINDOW_SECONDS": "0"}
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert any("RATE_LIMIT_WINDOW_SECONDS" in e for e in errors)


def test_validate_invalid_concurrency():
    """Engine max concurrency < 1 produces an error."""
    env = {"ENGINE_MAX_CONCURRENCY": "0"}
    with patch.dict(os.environ, env):
        config = AppConfig()
    errors = config.validate()
    assert any("ENGINE_MAX_CONCURRENCY" in e for e in errors)


# ---------------------------------------------------------------------------
# GET /api/v1/config
# ---------------------------------------------------------------------------


def test_config_endpoint_returns_200(client):
    """GET /config returns 200."""
    resp = client.get("/api/v1/config")
    assert resp.status_code == 200


def test_config_endpoint_has_expected_keys(client):
    """GET /config response includes core config keys."""
    resp = client.get("/api/v1/config")
    data = resp.json()
    assert "backend_port" in data
    assert "production" in data
    assert "debug" in data
    assert "log_level" in data
    assert "rate_limit_free" in data
    assert "engine_max_concurrency" in data
    assert "memory_backend" in data


def test_config_endpoint_redacts_secrets(client):
    """GET /config redacts secret values."""
    resp = client.get("/api/v1/config")
    data = resp.json()
    # jwt_secret_key should be redacted
    jwt_val = data.get("jwt_secret_key", "")
    if jwt_val:
        assert "***" in jwt_val


def test_config_endpoint_has_validation_errors(client):
    """GET /config includes _validation_errors list."""
    resp = client.get("/api/v1/config")
    data = resp.json()
    assert "_validation_errors" in data
    assert isinstance(data["_validation_errors"], list)


def test_config_endpoint_has_env_file_loaded(client):
    """GET /config includes _env_file_loaded field."""
    resp = client.get("/api/v1/config")
    data = resp.json()
    assert "_env_file_loaded" in data


# ---------------------------------------------------------------------------
# .env loading verification
# ---------------------------------------------------------------------------


def test_dotenv_loaded():
    """Verify that dotenv loaded environment variables."""
    # The main module imports and calls load_dotenv at module level.
    # If .env or .env.development exists, BACKEND_CORS_ORIGINS should be set.
    from apps.orchestrator.main import env_path
    if env_path.exists():
        # The env file was loaded — CORS origins should be set
        cors = os.environ.get("BACKEND_CORS_ORIGINS", "")
        assert cors  # should be non-empty if .env loaded
    else:
        pytest.skip("No .env or .env.development file found")
