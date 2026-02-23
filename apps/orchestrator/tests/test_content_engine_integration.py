"""
DIRECTIVE-NXTG-20260222-03: Content-Engine Workflow Template Validation

End-to-end integration test for the Content Engine Pipeline workflow template.
Validates: Start → HTTP (Research) → LLM (Enrich) → Code (Format) → Memory (Store) → End

HTTP and LLM nodes are mocked (no external APIs in CI). Code and Memory nodes
execute for real, validating the actual data flow through the pipeline.
"""
import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
import yaml

from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.main import (
    AppletMessage,
    Orchestrator,
    app,
)
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository


# ============================================================
# The Content Engine Pipeline (mirrors ContentEngine.ts + content_engine.yaml)
# ============================================================
CONTENT_ENGINE_FLOW = {
    "id": "content-engine-pipeline-validation",
    "name": "Content Engine Pipeline",
    "nodes": [
        {
            "id": "start",
            "type": "start",
            "position": {"x": 300, "y": 25},
            "data": {"label": "Research Topic"},
        },
        {
            "id": "research",
            "type": "http_request",
            "position": {"x": 300, "y": 150},
            "data": {
                "label": "Fetch Source",
                "method": "GET",
                "url": "{{input.url}}",
                "headers": {"Accept": "application/json"},
                "timeout_seconds": 30,
            },
        },
        {
            "id": "enrich",
            "type": "llm",
            "position": {"x": 300, "y": 300},
            "data": {
                "label": "Summarize Content",
                "provider": "ollama",
                "model": "llama3.1",
                "base_url": "http://localhost:11434",
                "temperature": 0.3,
                "max_tokens": 1024,
                "system_prompt": (
                    "You are a research assistant for a content engine. "
                    "Summarize the provided content into clear, concise key points. "
                    "Focus on facts, insights, and actionable information. "
                    "Output a structured summary with bullet points."
                ),
            },
        },
        {
            "id": "format",
            "type": "code",
            "position": {"x": 300, "y": 450},
            "data": {
                "label": "Format Article",
                "language": "python",
                "timeout_seconds": 5,
                "memory_limit_mb": 256,
                "code": (
                    "import json\n"
                    "import datetime\n"
                    "\n"
                    "# data = LLM summary output\n"
                    '# context["input"] = original user input with topic/url\n'
                    'raw_input = context.get("input", {})\n'
                    "if isinstance(raw_input, dict):\n"
                    '    topic = raw_input.get("topic", "Untitled")\n'
                    '    source_url = raw_input.get("url", "")\n'
                    "else:\n"
                    "    topic = str(raw_input)\n"
                    '    source_url = ""\n'
                    "\n"
                    'summary = str(data).strip() if data else "No summary available."\n'
                    "\n"
                    'article = f"# {topic}\\n\\n"\n'
                    "article += f\"*Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\\n\\n\"\n"
                    "if source_url:\n"
                    '    article += f"**Source:** {source_url}\\n\\n"\n'
                    'article += "---\\n\\n"\n'
                    'article += f"## Summary\\n\\n{summary}\\n"\n'
                    "\n"
                    "result = {\n"
                    '    "title": topic,\n'
                    '    "content": article,\n'
                    '    "source_url": source_url,\n'
                    '    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",\n'
                    '    "format": "markdown",\n'
                    "}\n"
                ),
            },
        },
        {
            "id": "store",
            "type": "memory",
            "position": {"x": 300, "y": 600},
            "data": {
                "label": "Store Article",
                "operation": "store",
                "key": "content-engine-article",
                "namespace": "content-engine",
            },
        },
        {
            "id": "end",
            "type": "end",
            "position": {"x": 300, "y": 720},
            "data": {"label": "Published Article"},
        },
    ],
    "edges": [
        {"id": "start-research", "source": "start", "target": "research"},
        {"id": "research-enrich", "source": "research", "target": "enrich"},
        {"id": "enrich-format", "source": "enrich", "target": "format"},
        {"id": "format-store", "source": "format", "target": "store"},
        {"id": "store-end", "source": "store", "target": "end"},
    ],
}


# ============================================================
# Fixtures
# ============================================================

@pytest_asyncio.fixture(scope="function")
async def db():
    await init_db()
    yield
    await close_db_connections()


# ============================================================
# Tests
# ============================================================

class TestContentEngineTemplateStructure:
    """Validate the template definition is well-formed."""

    def test_template_has_correct_node_count(self):
        assert len(CONTENT_ENGINE_FLOW["nodes"]) == 6, "Expected 6 nodes"

    def test_template_has_correct_edge_count(self):
        assert len(CONTENT_ENGINE_FLOW["edges"]) == 5, "Expected 5 edges"

    def test_template_has_required_node_types(self):
        node_types = {n["type"] for n in CONTENT_ENGINE_FLOW["nodes"]}
        assert node_types == {"start", "http_request", "llm", "code", "memory", "end"}

    def test_all_edges_reference_valid_nodes(self):
        node_ids = {n["id"] for n in CONTENT_ENGINE_FLOW["nodes"]}
        for edge in CONTENT_ENGINE_FLOW["edges"]:
            assert edge["source"] in node_ids, f"Edge source '{edge['source']}' not in nodes"
            assert edge["target"] in node_ids, f"Edge target '{edge['target']}' not in nodes"

    def test_pipeline_order_is_linear(self):
        """Verify edges form a linear chain: start → research → enrich → format → store → end."""
        expected_chain = ["start", "research", "enrich", "format", "store", "end"]
        edges = CONTENT_ENGINE_FLOW["edges"]
        for i, edge in enumerate(edges):
            assert edge["source"] == expected_chain[i]
            assert edge["target"] == expected_chain[i + 1]

    def test_yaml_template_loads_and_matches(self):
        """Verify the YAML template file loads and has matching structure."""
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "templates", "content_engine.yaml"
        )
        assert os.path.exists(yaml_path), f"YAML template not found at {yaml_path}"

        with open(yaml_path) as f:
            tmpl = yaml.safe_load(f)

        assert tmpl["id"] == "content-engine-pipeline"
        assert tmpl["name"] == "Content Engine Pipeline"
        assert len(tmpl["nodes"]) == 6
        assert len(tmpl["edges"]) == 5

        yaml_node_types = {n["type"] for n in tmpl["nodes"]}
        assert yaml_node_types == {"start", "http_request", "llm", "code", "memory", "end"}


class TestContentEngineIntegration:
    """End-to-end validation of the Content Engine Pipeline workflow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, db):
        """Run the complete pipeline with mocked HTTP and LLM responses."""
        mock_http_response = AppletMessage(
            content={
                "data": "Python 3.12 introduces several performance improvements...",
                "status_code": 200,
            },
            context={},
            metadata={"applet": "http_request", "status": "success", "status_code": 200},
        )
        mock_llm_response = AppletMessage(
            content=(
                "- Python 3.12 brings significant performance gains\n"
                "- New type parameter syntax simplifies generics\n"
                "- Improved error messages for debugging"
            ),
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.HTTPRequestNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_http_response,
        ):
            with patch(
                "apps.orchestrator.main.LLMNodeApplet.on_message",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ):
                with patch(
                    "apps.orchestrator.main.broadcast_status", new_callable=AsyncMock
                ):
                    run_id = await Orchestrator.execute_flow(
                        CONTENT_ENGINE_FLOW,
                        {
                            "topic": "Python 3.12 Features",
                            "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
                        },
                    )

                    assert run_id is not None

                    run = None
                    for _ in range(40):
                        await asyncio.sleep(0.15)
                        run = await WorkflowRunRepository.get_by_run_id(run_id)
                        if run and run.get("status") in ("success", "error"):
                            break

                    assert run is not None, "Run record not found"
                    assert run["status"] == "success", (
                        f"Expected success but got {run['status']}. "
                        f"Error: {run.get('error', 'N/A')}. "
                        f"Details: {run.get('error_details', 'N/A')}"
                    )

                    results = run.get("results", {})
                    assert "research" in results, "HTTP research node did not produce results"
                    assert "enrich" in results, "LLM enrich node did not produce results"
                    assert "format" in results, "Code format node did not produce results"
                    assert "store" in results, "Memory store node did not produce results"

    @pytest.mark.asyncio
    async def test_pipeline_with_empty_summary(self, db):
        """Pipeline should handle empty LLM output gracefully."""
        mock_http_response = AppletMessage(
            content={"data": "Some content", "status_code": 200},
            context={},
            metadata={"applet": "http_request", "status": "success", "status_code": 200},
        )
        mock_llm_response = AppletMessage(
            content="",
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.HTTPRequestNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_http_response,
        ):
            with patch(
                "apps.orchestrator.main.LLMNodeApplet.on_message",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ):
                with patch(
                    "apps.orchestrator.main.broadcast_status", new_callable=AsyncMock
                ):
                    run_id = await Orchestrator.execute_flow(
                        CONTENT_ENGINE_FLOW,
                        {"topic": "Empty Test", "url": "https://example.com"},
                    )

                    run = None
                    for _ in range(40):
                        await asyncio.sleep(0.15)
                        run = await WorkflowRunRepository.get_by_run_id(run_id)
                        if run and run.get("status") in ("success", "error"):
                            break

                    assert run is not None
                    assert run["status"] == "success", (
                        f"Expected success but got {run['status']}. "
                        f"Error: {run.get('error', 'N/A')}"
                    )

    @pytest.mark.asyncio
    async def test_pipeline_via_api_endpoint(self, db):
        """Validate the flow can be created and run via the REST API."""
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.post("/api/v1/flows", json=CONTENT_ENGINE_FLOW)
            assert resp.status_code == 201, f"Flow creation failed: {resp.text}"

            resp = client.get(f"/api/v1/flows/{CONTENT_ENGINE_FLOW['id']}")
            assert resp.status_code == 200
            flow_data = resp.json()
            assert flow_data["name"] == "Content Engine Pipeline"
            assert len(flow_data["nodes"]) == 6
            assert len(flow_data["edges"]) == 5

            mock_http_response = AppletMessage(
                content={"data": "Test data", "status_code": 200},
                context={},
                metadata={"applet": "http_request", "status": "success"},
            )
            mock_llm_response = AppletMessage(
                content="Summary of test data",
                context={},
                metadata={"applet": "llm", "status": "success"},
            )
            with patch(
                "apps.orchestrator.main.HTTPRequestNodeApplet.on_message",
                new_callable=AsyncMock,
                return_value=mock_http_response,
            ):
                with patch(
                    "apps.orchestrator.main.LLMNodeApplet.on_message",
                    new_callable=AsyncMock,
                    return_value=mock_llm_response,
                ):
                    with patch(
                        "apps.orchestrator.main.broadcast_status", new_callable=AsyncMock
                    ):
                        resp = client.post(
                            f"/api/v1/flows/{CONTENT_ENGINE_FLOW['id']}/runs",
                            json={
                                "input": {
                                    "topic": "API Test",
                                    "url": "https://example.com/api",
                                }
                            },
                        )
                        assert resp.status_code == 202, f"Run failed: {resp.text}"
                        assert "run_id" in resp.json()
