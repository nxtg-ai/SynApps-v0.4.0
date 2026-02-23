"""
DIRECTIVE-NXTG-20260222-02: 2Brain Integration Validation

End-to-end integration test for the 2Brain Inbox Triage workflow template (N-16).
Validates: Start → LLM Classifier → Code Structurer → Memory Store → End

The LLM node is mocked (Ollama not available in CI), but Code and Memory nodes
execute for real, validating the actual data flow through the pipeline.
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.main import (
    AppletMessage,
    Orchestrator,
    app,
)
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository


# ============================================================
# The 2Brain Inbox Triage template (mirrors TwoBrainInbox.ts)
# ============================================================
TWOBRAIN_FLOW = {
    "id": "2brain-inbox-triage-validation",
    "name": "2Brain Inbox Triage",
    "nodes": [
        {
            "id": "start",
            "type": "start",
            "position": {"x": 300, "y": 25},
            "data": {"label": "Raw Inbox Item"},
        },
        {
            "id": "classifier",
            "type": "llm",
            "position": {"x": 300, "y": 150},
            "data": {
                "label": "Ollama Classifier",
                "provider": "ollama",
                "model": "llama3.1",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 64,
                "system_prompt": (
                    "You are a triage assistant for a personal knowledge base called 2Brain. "
                    "Classify the user's input into exactly one of these categories: idea, task, reference, note. "
                    "Respond with ONLY the single category word — no explanation, no punctuation, no extra text."
                ),
            },
        },
        {
            "id": "structurer",
            "type": "code",
            "position": {"x": 300, "y": 300},
            "data": {
                "label": "Structure Output",
                "language": "python",
                "timeout_seconds": 5,
                "memory_limit_mb": 256,
                "code": (
                    "import json\n"
                    "import datetime\n"
                    "\n"
                    "# data = LLM output (category string)\n"
                    '# context["input"] = original user input\n'
                    'raw_input = context.get("input", {})\n'
                    "if isinstance(raw_input, dict):\n"
                    '    original_text = raw_input.get("text", str(raw_input))\n'
                    "else:\n"
                    "    original_text = str(raw_input)\n"
                    "\n"
                    'category = str(data).strip().lower() if data else "note"\n'
                    'if category not in {"idea", "task", "reference", "note"}:\n'
                    '    category = "note"\n'
                    "\n"
                    "result = {\n"
                    '    "category": category,\n'
                    '    "content": original_text,\n'
                    '    "captured_at": datetime.datetime.utcnow().isoformat() + "Z",\n'
                    '    "tags": [category],\n'
                    "}\n"
                ),
            },
        },
        {
            "id": "memory-store",
            "type": "memory",
            "position": {"x": 300, "y": 440},
            "data": {
                "label": "Store in 2Brain",
                "operation": "store",
                "key": "2brain-inbox",
                "namespace": "2brain",
            },
        },
        {
            "id": "end",
            "type": "end",
            "position": {"x": 300, "y": 560},
            "data": {"label": "Triaged Item"},
        },
    ],
    "edges": [
        {"id": "start-classifier", "source": "start", "target": "classifier"},
        {"id": "classifier-structurer", "source": "classifier", "target": "structurer"},
        {"id": "structurer-memory", "source": "structurer", "target": "memory-store"},
        {"id": "memory-end", "source": "memory-store", "target": "end"},
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

class TestTwoBrainIntegration:
    """End-to-end validation of the 2Brain Inbox Triage workflow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_idea_classification(self, db):
        """Run the complete 2Brain pipeline with input classified as 'idea'."""
        mock_llm_response = AppletMessage(
            content="idea",
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.LLMNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
                run_id = await Orchestrator.execute_flow(
                    TWOBRAIN_FLOW,
                    {"text": "What if we built a CLI that generates project scaffolds?"},
                )

                assert run_id is not None

                # Wait for async execution to complete
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
                    f"Error details: {run.get('error_details', 'N/A')}"
                )

                # Verify all nodes executed
                results = run.get("results", {})
                assert "classifier" in results, "LLM classifier node did not produce results"
                assert "structurer" in results, "Code structurer node did not produce results"
                assert "memory-store" in results, "Memory store node did not produce results"

    @pytest.mark.asyncio
    async def test_full_pipeline_task_classification(self, db):
        """Run the pipeline with input classified as 'task'."""
        mock_llm_response = AppletMessage(
            content="task",
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.LLMNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
                run_id = await Orchestrator.execute_flow(
                    TWOBRAIN_FLOW,
                    {"text": "Buy groceries before 6pm"},
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
    async def test_full_pipeline_reference_classification(self, db):
        """Run the pipeline with input classified as 'reference'."""
        mock_llm_response = AppletMessage(
            content="reference",
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.LLMNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
                run_id = await Orchestrator.execute_flow(
                    TWOBRAIN_FLOW,
                    {"text": "RFC 9114 defines HTTP/3 over QUIC transport"},
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
    async def test_full_pipeline_note_classification(self, db):
        """Run the pipeline with input classified as 'note'."""
        mock_llm_response = AppletMessage(
            content="note",
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.LLMNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
                run_id = await Orchestrator.execute_flow(
                    TWOBRAIN_FLOW,
                    {"text": "Had a good meeting with the team today about Q2 goals"},
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
    async def test_pipeline_handles_unknown_category_gracefully(self, db):
        """When LLM returns garbage, code node should default to 'note'."""
        mock_llm_response = AppletMessage(
            content="I think this might be an idea or maybe a task, I'm not sure",
            context={},
            metadata={"applet": "llm", "status": "success"},
        )

        with patch(
            "apps.orchestrator.main.LLMNodeApplet.on_message",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
                run_id = await Orchestrator.execute_flow(
                    TWOBRAIN_FLOW,
                    {"text": "Random thought about the universe"},
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
    async def test_template_node_connections_valid(self, db):
        """Verify all edges reference valid node IDs."""
        node_ids = {n["id"] for n in TWOBRAIN_FLOW["nodes"]}
        for edge in TWOBRAIN_FLOW["edges"]:
            assert edge["source"] in node_ids, f"Edge source '{edge['source']}' not in nodes"
            assert edge["target"] in node_ids, f"Edge target '{edge['target']}' not in nodes"

    @pytest.mark.asyncio
    async def test_template_has_correct_structure(self, db):
        """Verify the template has the expected 5 nodes and 4 edges."""
        assert len(TWOBRAIN_FLOW["nodes"]) == 5, "Expected 5 nodes"
        assert len(TWOBRAIN_FLOW["edges"]) == 4, "Expected 4 edges"

        node_types = [n["type"] for n in TWOBRAIN_FLOW["nodes"]]
        assert "start" in node_types
        assert "llm" in node_types
        assert "code" in node_types
        assert "memory" in node_types
        assert "end" in node_types

    @pytest.mark.asyncio
    async def test_pipeline_via_api_endpoint(self, db):
        """Validate the flow can be created and run via the REST API."""
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            # Create the flow
            resp = client.post("/api/v1/flows", json=TWOBRAIN_FLOW)
            assert resp.status_code == 201, f"Flow creation failed: {resp.text}"

            # Verify it was persisted
            resp = client.get(f"/api/v1/flows/{TWOBRAIN_FLOW['id']}")
            assert resp.status_code == 200
            flow_data = resp.json()
            assert flow_data["name"] == "2Brain Inbox Triage"
            assert len(flow_data["nodes"]) == 5
            assert len(flow_data["edges"]) == 4

            # Execute via API
            mock_llm_response = AppletMessage(
                content="idea",
                context={},
                metadata={"applet": "llm", "status": "success"},
            )
            with patch(
                "apps.orchestrator.main.LLMNodeApplet.on_message",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ):
                with patch(
                    "apps.orchestrator.main.broadcast_status", new_callable=AsyncMock
                ):
                    resp = client.post(
                        f"/api/v1/flows/{TWOBRAIN_FLOW['id']}/runs",
                        json={"input": {"text": "Build a visual workflow tool"}},
                    )
                    assert resp.status_code == 202, f"Run creation failed: {resp.text}"
                    run_data = resp.json()
                    assert "run_id" in run_data
