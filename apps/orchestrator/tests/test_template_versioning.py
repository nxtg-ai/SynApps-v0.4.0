"""Tests for Template Versioning — semver, fetch-by-version, rollback (DIRECTIVE-09)."""

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    app,
    template_registry,
    _parse_semver,
    _bump_patch,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_templates():
    template_registry.reset()
    yield
    template_registry.reset()


SAMPLE = {
    "id": "my-template",
    "name": "My Template",
    "description": "Test template",
    "tags": ["test"],
    "nodes": [
        {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
        {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
    ],
    "edges": [{"id": "e1", "source": "start", "target": "end"}],
}


# ---------------------------------------------------------------------------
# _parse_semver / _bump_patch helpers
# ---------------------------------------------------------------------------


class TestSemverHelpers:
    def test_parse_valid(self):
        assert _parse_semver("1.2.3") == (1, 2, 3)

    def test_parse_zero(self):
        assert _parse_semver("0.0.0") == (0, 0, 0)

    def test_parse_large(self):
        assert _parse_semver("100.200.300") == (100, 200, 300)

    def test_parse_invalid_alpha(self):
        assert _parse_semver("abc") is None

    def test_parse_invalid_two_parts(self):
        assert _parse_semver("1.2") is None

    def test_parse_invalid_suffix(self):
        assert _parse_semver("1.2.3-beta") is None

    def test_bump_patch(self):
        assert _bump_patch("1.0.0") == "1.0.1"

    def test_bump_patch_increments(self):
        assert _bump_patch("2.5.9") == "2.5.10"

    def test_bump_patch_invalid_returns_default(self):
        assert _bump_patch("bad") == "1.0.0"


# ---------------------------------------------------------------------------
# TemplateRegistry — semver unit tests
# ---------------------------------------------------------------------------


class TestRegistrySemver:
    def test_first_import_gets_1_0_0(self):
        entry = template_registry.import_template(SAMPLE)
        assert entry["semver"] == "1.0.0"

    def test_second_import_auto_bumps_patch(self):
        template_registry.import_template(SAMPLE)
        entry = template_registry.import_template({**SAMPLE, "name": "V2"})
        assert entry["semver"] == "1.0.1"

    def test_third_import_bumps_again(self):
        template_registry.import_template(SAMPLE)
        template_registry.import_template({**SAMPLE, "name": "V2"})
        entry = template_registry.import_template({**SAMPLE, "name": "V3"})
        assert entry["semver"] == "1.0.2"

    def test_explicit_semver_accepted(self):
        entry = template_registry.import_template({**SAMPLE, "version": "2.0.0"})
        assert entry["semver"] == "2.0.0"

    def test_explicit_semver_minor_bump(self):
        template_registry.import_template(SAMPLE)
        entry = template_registry.import_template({**SAMPLE, "version": "1.1.0"})
        assert entry["semver"] == "1.1.0"

    def test_duplicate_semver_raises(self):
        template_registry.import_template(SAMPLE)  # 1.0.0
        with pytest.raises(ValueError, match="already exists"):
            template_registry.import_template({**SAMPLE, "version": "1.0.0"})

    def test_invalid_semver_raises(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            template_registry.import_template({**SAMPLE, "version": "bad"})

    def test_get_by_semver_latest(self):
        template_registry.import_template(SAMPLE)
        template_registry.import_template({**SAMPLE, "name": "V2"})
        result = template_registry.get_by_semver("my-template")
        assert result["name"] == "V2"
        assert result["semver"] == "1.0.1"

    def test_get_by_semver_specific(self):
        template_registry.import_template(SAMPLE)
        template_registry.import_template({**SAMPLE, "name": "V2"})
        result = template_registry.get_by_semver("my-template", semver="1.0.0")
        assert result["name"] == "My Template"
        assert result["semver"] == "1.0.0"

    def test_get_by_semver_not_found(self):
        template_registry.import_template(SAMPLE)
        assert template_registry.get_by_semver("my-template", semver="9.9.9") is None

    def test_get_by_semver_unknown_template(self):
        assert template_registry.get_by_semver("nonexistent") is None


# ---------------------------------------------------------------------------
# TemplateRegistry — rollback unit tests
# ---------------------------------------------------------------------------


class TestRegistryRollback:
    def test_rollback_creates_new_version(self):
        template_registry.import_template(SAMPLE)
        template_registry.import_template({**SAMPLE, "name": "V2"})
        entry = template_registry.rollback("my-template", "1.0.0")
        assert entry is not None
        assert entry["version"] == 3
        assert entry["name"] == "My Template"  # restored from v1
        assert entry["rolled_back_from"] == "1.0.0"
        assert entry["semver"] == "1.0.2"  # auto-bumped from 1.0.1

    def test_rollback_preserves_original(self):
        template_registry.import_template(SAMPLE)
        template_registry.import_template({**SAMPLE, "name": "V2"})
        template_registry.rollback("my-template", "1.0.0")
        # Original versions still exist
        versions = template_registry.list_versions("my-template")
        assert len(versions) == 3
        assert versions[0]["name"] == "My Template"
        assert versions[1]["name"] == "V2"
        assert versions[2]["name"] == "My Template"

    def test_rollback_unknown_template(self):
        assert template_registry.rollback("nonexistent", "1.0.0") is None

    def test_rollback_unknown_version(self):
        template_registry.import_template(SAMPLE)
        assert template_registry.rollback("my-template", "9.9.9") is None

    def test_rollback_nodes_and_edges_copied(self):
        template_registry.import_template(SAMPLE)
        template_registry.import_template({
            **SAMPLE,
            "name": "V2",
            "nodes": [{"id": "only-node", "type": "llm", "position": {"x": 0, "y": 0}, "data": {}}],
            "edges": [],
        })
        entry = template_registry.rollback("my-template", "1.0.0")
        assert len(entry["nodes"]) == 2  # from v1
        assert len(entry["edges"]) == 1  # from v1


# ---------------------------------------------------------------------------
# GET /api/v1/templates/{id}/by-semver — fetch by semver
# ---------------------------------------------------------------------------


class TestGetBySemverEndpoint:
    def test_fetch_latest(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        client.post("/api/v1/templates/import", json={**SAMPLE, "name": "V2"})
        resp = client.get("/api/v1/templates/my-template/by-semver")
        assert resp.status_code == 200
        assert resp.json()["name"] == "V2"
        assert resp.json()["semver"] == "1.0.1"

    def test_fetch_specific_version(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        client.post("/api/v1/templates/import", json={**SAMPLE, "name": "V2"})
        resp = client.get("/api/v1/templates/my-template/by-semver?version=1.0.0")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Template"
        assert resp.json()["semver"] == "1.0.0"

    def test_fetch_version_not_found(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        resp = client.get("/api/v1/templates/my-template/by-semver?version=9.9.9")
        assert resp.status_code == 404
        assert "9.9.9" in resp.json()["error"]["message"]

    def test_fetch_template_not_found(self, client):
        resp = client.get("/api/v1/templates/nonexistent/by-semver")
        assert resp.status_code == 404

    def test_fetch_invalid_semver_rejected(self, client):
        resp = client.get("/api/v1/templates/my-template/by-semver?version=bad")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# PUT /api/v1/templates/{id}/rollback — rollback endpoint
# ---------------------------------------------------------------------------


class TestRollbackEndpoint:
    def test_rollback_success(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        client.post("/api/v1/templates/import", json={**SAMPLE, "name": "V2"})
        resp = client.put("/api/v1/templates/my-template/rollback?version=1.0.0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 3
        assert data["name"] == "My Template"
        assert data["rolled_back_from"] == "1.0.0"
        assert data["semver"] == "1.0.2"

    def test_rollback_template_not_found(self, client):
        resp = client.put("/api/v1/templates/nonexistent/rollback?version=1.0.0")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]["message"]

    def test_rollback_version_not_found(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        resp = client.put("/api/v1/templates/my-template/rollback?version=9.9.9")
        assert resp.status_code == 404
        assert "9.9.9" in resp.json()["error"]["message"]

    def test_rollback_missing_version_param(self, client):
        resp = client.put("/api/v1/templates/my-template/rollback")
        assert resp.status_code == 422

    def test_rollback_invalid_semver_rejected(self, client):
        resp = client.put("/api/v1/templates/my-template/rollback?version=bad")
        assert resp.status_code == 422

    def test_rollback_latest_reflects_rollback(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        client.post("/api/v1/templates/import", json={**SAMPLE, "name": "V2"})
        client.put("/api/v1/templates/my-template/rollback?version=1.0.0")
        # Latest should now be the rolled-back version
        resp = client.get("/api/v1/templates/my-template/by-semver")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Template"
        assert resp.json()["semver"] == "1.0.2"


# ---------------------------------------------------------------------------
# POST /api/v1/templates/import — semver in import
# ---------------------------------------------------------------------------


class TestImportWithSemver:
    def test_import_explicit_semver(self, client):
        resp = client.post("/api/v1/templates/import", json={**SAMPLE, "version": "2.0.0"})
        assert resp.status_code == 201
        assert resp.json()["semver"] == "2.0.0"

    def test_import_auto_semver(self, client):
        resp = client.post("/api/v1/templates/import", json=SAMPLE)
        assert resp.status_code == 201
        assert resp.json()["semver"] == "1.0.0"

    def test_import_duplicate_semver_409(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        resp = client.post("/api/v1/templates/import", json={**SAMPLE, "version": "1.0.0"})
        assert resp.status_code == 409
        assert "already exists" in resp.json()["error"]["message"]

    def test_import_invalid_semver_422(self, client):
        resp = client.post("/api/v1/templates/import", json={**SAMPLE, "version": "xyz"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration: versions endpoint includes semver
# ---------------------------------------------------------------------------


class TestVersionsIncludeSemver:
    def test_versions_have_semver(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        client.post("/api/v1/templates/import", json={**SAMPLE, "name": "V2"})
        resp = client.get("/api/v1/templates/my-template/versions")
        assert resp.status_code == 200
        versions = resp.json()["versions"]
        assert versions[0]["semver"] == "1.0.0"
        assert versions[1]["semver"] == "1.0.1"

    def test_list_templates_has_semver(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        resp = client.get("/api/v1/templates")
        assert resp.status_code == 200
        assert resp.json()["templates"][0]["semver"] == "1.0.0"

    def test_export_has_semver(self, client):
        client.post("/api/v1/templates/import", json=SAMPLE)
        resp = client.get("/api/v1/templates/my-template/export")
        assert resp.status_code == 200
        assert resp.json()["semver"] == "1.0.0"
