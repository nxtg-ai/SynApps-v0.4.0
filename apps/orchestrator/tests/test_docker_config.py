"""
CI-safe tests for Docker build configuration.

These tests validate Dockerfile and docker-compose structure/syntax
WITHOUT requiring a Docker daemon. They parse the files and verify:
- Required stages, instructions, and best practices in Dockerfiles
- Service definitions, health checks, and networks in docker-compose
- .dockerignore covers sensitive files
"""

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]  # repo root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(rel_path: str) -> str:
    path = ROOT / rel_path
    assert path.exists(), f"Missing expected file: {rel_path}"
    return path.read_text()


def _parse_compose(rel_path: str) -> dict:
    return yaml.safe_load(_read(rel_path))


# ---------------------------------------------------------------------------
# Dockerfile.orchestrator
# ---------------------------------------------------------------------------

class TestDockerfileOrchestrator:
    """Verify multi-stage build structure and best practices."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("infra/docker/Dockerfile.orchestrator")

    def test_has_builder_stage(self):
        assert re.search(r"FROM\s+python:.*\s+AS\s+builder", self.content)

    def test_has_runtime_stage(self):
        assert re.search(r"FROM\s+python:.*\s+AS\s+runtime", self.content)

    def test_copies_from_builder(self):
        assert "COPY --from=builder" in self.content

    def test_non_root_user(self):
        assert "USER synapps" in self.content

    def test_healthcheck_present(self):
        assert "HEALTHCHECK" in self.content

    def test_exposes_port_8000(self):
        assert "EXPOSE 8000" in self.content

    def test_copies_requirements_first(self):
        """requirements.txt should be copied before app source for layer caching."""
        req_pos = self.content.find("requirements.txt")
        app_pos = self.content.find("COPY apps/orchestrator /app")
        assert req_pos < app_pos, "requirements.txt should be copied before app source"

    def test_no_pip_cache(self):
        assert "PIP_NO_CACHE_DIR" in self.content

    def test_uvicorn_entrypoint(self):
        assert "uvicorn" in self.content
        assert "apps.orchestrator.main:app" in self.content


# ---------------------------------------------------------------------------
# Dockerfile.frontend
# ---------------------------------------------------------------------------

class TestDockerfileFrontend:

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = ROOT / "infra/docker/Dockerfile.frontend"
        if not self.path.exists():
            pytest.skip("Dockerfile.frontend not present")
        self.content = self.path.read_text()

    def test_has_build_stage(self):
        assert re.search(r"FROM\s+node:.*\s+AS\s+\S+build\S*", self.content, re.IGNORECASE)

    def test_has_production_stage(self):
        assert re.search(r"FROM\s+(nginx|node)", self.content)

    def test_exposes_port(self):
        assert "EXPOSE" in self.content


# ---------------------------------------------------------------------------
# docker-compose.yml (root)
# ---------------------------------------------------------------------------

class TestRootDockerCompose:

    @pytest.fixture(autouse=True)
    def _load(self):
        self.compose = _parse_compose("docker-compose.yml")

    def test_has_db_service(self):
        assert "db" in self.compose["services"]

    def test_has_orchestrator_service(self):
        assert "orchestrator" in self.compose["services"]

    def test_has_frontend_service(self):
        assert "frontend" in self.compose["services"]

    def test_db_is_postgres(self):
        db = self.compose["services"]["db"]
        assert "postgres" in db["image"]

    def test_orchestrator_depends_on_db(self):
        orch = self.compose["services"]["orchestrator"]
        assert "db" in orch["depends_on"]

    def test_frontend_depends_on_orchestrator(self):
        fe = self.compose["services"]["frontend"]
        assert "orchestrator" in fe["depends_on"]

    def test_all_services_have_healthcheck(self):
        for name, svc in self.compose["services"].items():
            assert "healthcheck" in svc, f"Service '{name}' missing healthcheck"

    def test_all_services_on_same_network(self):
        for name, svc in self.compose["services"].items():
            networks = svc.get("networks", [])
            assert "synapps-network" in networks, f"Service '{name}' not on synapps-network"

    def test_postgres_volume_defined(self):
        assert "postgres_data" in self.compose.get("volumes", {})

    def test_orchestrator_has_database_url(self):
        env = self.compose["services"]["orchestrator"]["environment"]
        db_url = env.get("DATABASE_URL", "")
        assert "postgresql" in db_url

    def test_orchestrator_has_jwt_secret(self):
        env = self.compose["services"]["orchestrator"]["environment"]
        assert "JWT_SECRET_KEY" in env

    def test_orchestrator_has_rate_limit_vars(self):
        env = self.compose["services"]["orchestrator"]["environment"]
        assert "RATE_LIMIT_WINDOW_SECONDS" in env
        assert "RATE_LIMIT_FREE" in env

    def test_frontend_has_api_url(self):
        env = self.compose["services"]["frontend"]["environment"]
        assert "VITE_API_URL" in env

    def test_restart_policies(self):
        for name, svc in self.compose["services"].items():
            assert svc.get("restart") == "unless-stopped", (
                f"Service '{name}' should have restart: unless-stopped"
            )


# ---------------------------------------------------------------------------
# infra/docker/docker-compose.yml
# ---------------------------------------------------------------------------

class TestInfraDockerCompose:

    @pytest.fixture(autouse=True)
    def _load(self):
        self.compose = _parse_compose("infra/docker/docker-compose.yml")

    def test_has_same_services_as_root(self):
        root = _parse_compose("docker-compose.yml")
        assert set(self.compose["services"].keys()) == set(root["services"].keys())

    def test_build_context_is_parent(self):
        """infra/docker compose should use ../../ as build context."""
        orch = self.compose["services"]["orchestrator"]
        assert orch["build"]["context"] == "../../"


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

class TestDockerignore:

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(".dockerignore")

    def test_ignores_git(self):
        assert ".git" in self.content

    def test_ignores_pycache(self):
        assert "__pycache__" in self.content

    def test_ignores_node_modules(self):
        assert "node_modules" in self.content

    def test_ignores_env_files(self):
        assert ".env" in self.content

    def test_ignores_venv(self):
        assert ".venv" in self.content or "venv" in self.content

    def test_ignores_db_files(self):
        assert "*.db" in self.content

    def test_ignores_test_artifacts(self):
        assert "test-results" in self.content
