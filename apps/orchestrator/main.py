"""
SynApps Orchestrator - Core Module

This is the lightweight microkernel that routes messages between applets in a
defined sequence. The orchestrator's job is purely to pass messages and data
between applets.
"""
import asyncio
import importlib
import json
import logging
import math
import os
import sys
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from pathlib import Path

# Load environment variables from .env.development file
from dotenv import load_dotenv

# Load .env.development file from the project root directory
project_root = Path(__file__).parent.parent.parent
dotenv_path = project_root / ".env.development"
load_dotenv(dotenv_path=dotenv_path)

from fastapi import (
    Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect,
    BackgroundTasks, APIRouter, Query, Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Import database modules
from contextlib import asynccontextmanager
from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository
from apps.orchestrator.models import FlowModel, FlowNodeModel, FlowEdgeModel, WorkflowRunStatusModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")

# ============================================================
# Constants
# ============================================================

API_VERSION = "1.0.0"
WS_AUTH_TOKEN = os.environ.get("WS_AUTH_TOKEN")

# ============================================================
# Application Setup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database initialization and cleanup."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialization complete")
    yield
    logger.info("Closing database connections...")
    await close_db_connections()
    logger.info("Database connections closed")

app = FastAPI(title="SynApps Orchestrator", version=API_VERSION, lifespan=lifespan)

# Configure CORS
backend_cors_origins = os.environ.get("BACKEND_CORS_ORIGINS", "")
backend_cors_origins = backend_cors_origins.split(",")
allowed_origins = [origin.strip() for origin in backend_cors_origins if origin.strip()]

if not allowed_origins:
    logger.warning("No CORS origins specified, allowing all origins in development mode")
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Error Handling - Consistent Error Format
# ============================================================

_HTTP_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_SERVER_ERROR",
    501: "NOT_IMPLEMENTED",
    503: "SERVICE_UNAVAILABLE",
}


def _error_response(
    status: int,
    code: str,
    message: str,
    details: Optional[List[Dict[str, Any]]] = None,
) -> JSONResponse:
    """Create a standardized error JSONResponse."""
    body: Dict[str, Any] = {
        "error": {
            "code": code,
            "status": status,
            "message": message,
        }
    }
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = _HTTP_ERROR_CODES.get(exc.status_code, "ERROR")
    return _error_response(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        details.append({
            "field": field,
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        })
    return _error_response(422, "VALIDATION_ERROR", "Request validation failed", details)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return _error_response(500, "INTERNAL_SERVER_ERROR", "An unexpected error occurred")


# ============================================================
# Pagination
# ============================================================

def paginate(items: list, page: int, page_size: int) -> dict:
    """Apply offset-based pagination to a list of items."""
    total = len(items)
    total_pages = math.ceil(total / page_size) if page_size > 0 else 0
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


# ============================================================
# Request Validation Models (Pydantic v2 strict)
# ============================================================

class FlowNodeRequest(BaseModel):
    """Strictly validated flow node for API requests."""
    id: str = Field(..., min_length=1, max_length=200)
    type: str = Field(..., min_length=1, max_length=100)
    position: Dict[str, float]
    data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("position")
    @classmethod
    def validate_position(cls, v):
        if "x" not in v or "y" not in v:
            raise ValueError("Position must contain 'x' and 'y' keys")
        return v


class FlowEdgeRequest(BaseModel):
    """Strictly validated flow edge for API requests."""
    id: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    animated: bool = False


class CreateFlowRequest(BaseModel):
    """Strictly validated flow creation/update request."""
    id: Optional[str] = Field(None, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    nodes: List[FlowNodeRequest] = Field(default_factory=list)
    edges: List[FlowEdgeRequest] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v):
        if not v.strip():
            raise ValueError("Flow name cannot be blank")
        return v.strip()

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v):
        if v is not None and v.strip() == "":
            return None  # Treat blank as None so a UUID is auto-generated
        return v


class RunFlowRequest(BaseModel):
    """Strictly validated request body for running a flow."""
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")


class AISuggestRequest(BaseModel):
    """Strictly validated request body for AI suggestions."""
    prompt: str = Field(..., min_length=1, max_length=5000, description="The prompt for AI suggestion")
    context: Optional[str] = Field(None, max_length=10000, description="Optional context for the suggestion")


# ============================================================
# Internal Models
# ============================================================

class AppletStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"

class FlowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, int]
    data: Dict[str, Any] = Field(default_factory=dict)

class FlowEdge(BaseModel):
    id: str
    source: str
    target: str
    animated: bool = False

class Flow(BaseModel):
    id: str
    name: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]

class AppletMessage(BaseModel):
    content: Any
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowRunStatus(BaseModel):
    run_id: str
    flow_id: str
    status: str
    current_applet: Optional[str] = None
    progress: int = 0
    total_steps: int = 0
    start_time: float = 0
    end_time: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ============================================================
# WebSocket Protocol
# ============================================================

connected_clients: List[WebSocket] = []
ws_sessions: Dict[str, str] = {}  # session_id -> last known state for reconnection
applet_registry: Dict[str, Type['BaseApplet']] = {}


def _ws_message(msg_type: str, data: Optional[dict] = None) -> dict:
    """Create a structured WebSocket message with id, type, data, and timestamp."""
    return {
        "id": str(uuid.uuid4()),
        "type": msg_type,
        "data": data or {},
        "timestamp": time.time(),
    }


async def broadcast_status(status: Dict[str, Any]):
    """Broadcast workflow status to all connected clients using structured messages."""
    if not connected_clients:
        logger.warning("No connected clients to broadcast to")
        return

    broadcast_data = status.copy()
    if "completed_applets" not in broadcast_data:
        broadcast_data["completed_applets"] = []

    message = _ws_message("workflow.status", broadcast_data)

    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")
            disconnected.append(client)

    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)


# ============================================================
# Base Applet
# ============================================================

class BaseApplet:
    """Base class that all applets must implement."""

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return applet metadata."""
        return {
            "name": cls.__name__,
            "description": cls.__doc__ or "No description provided",
            "version": getattr(cls, "VERSION", "0.1.0"),
            "capabilities": getattr(cls, "CAPABILITIES", []),
        }

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message and return a response."""
        raise NotImplementedError("Applets must implement on_message")


# ============================================================
# Orchestrator Core
# ============================================================

class Orchestrator:
    """Core orchestration engine that executes applet flows."""

    @staticmethod
    async def load_applet(applet_type: str) -> BaseApplet:
        """Dynamically load an applet by type."""
        if applet_type in applet_registry:
            return applet_registry[applet_type]()

        try:
            module_path = f"apps.applets.{applet_type}.applet"
            module = importlib.import_module(module_path)
            applet_class = getattr(module, f"{applet_type.capitalize()}Applet")
            applet_registry[applet_type] = applet_class
            return applet_class()
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load applet '{applet_type}': {e}")
            raise ValueError(f"Applet type '{applet_type}' not found")

    @staticmethod
    def create_run_id() -> str:
        """Generate a unique run ID."""
        return str(uuid.uuid4())

    @staticmethod
    async def execute_flow(flow: dict, input_data: Dict[str, Any]) -> str:
        """Execute a flow and return the run ID."""
        run_id = Orchestrator.create_run_id()

        status = {
            "run_id": run_id,
            "flow_id": flow["id"],
            "status": "running",
            "current_applet": None,
            "progress": 0,
            "total_steps": len(flow["nodes"]),
            "start_time": time.time(),
            "results": {},
            "input_data": input_data
        }

        memory_completed_applets = []
        status_dict = status.copy()

        workflow_run_repo = WorkflowRunRepository()
        logger.info(f"Starting workflow execution with run ID: {run_id}")
        await workflow_run_repo.save(status_dict)

        broadcast_status_dict = status_dict.copy()
        broadcast_status_dict["completed_applets"] = []
        await broadcast_status(broadcast_status_dict)

        asyncio.create_task(
            Orchestrator._execute_flow_async(
                run_id, flow, input_data, workflow_run_repo, broadcast_status
            )
        )

        return run_id

    @staticmethod
    async def _execute_flow_async(
        run_id: str,
        flow: dict,
        input_data: Dict[str, Any],
        workflow_run_repo: WorkflowRunRepository,
        broadcast_status_fn,
    ):
        status = None
        memory_completed_applets = []

        try:
            status = await workflow_run_repo.get_by_run_id(run_id)

            nodes_by_id = {node["id"]: node for node in flow["nodes"]}

            graph = {}
            for edge in flow["edges"]:
                if edge["source"] not in graph:
                    graph[edge["source"]] = []
                graph[edge["source"]].append(edge["target"])

            target_nodes = set(edge["target"] for edge in flow["edges"])
            start_nodes = [node["id"] for node in flow["nodes"] if node["id"] not in target_nodes]

            if not start_nodes:
                if status and isinstance(status, dict):
                    status["status"] = "error"
                    status["error"] = "No start node found in flow"
                    status["end_time"] = time.time()
                    await workflow_run_repo.save(status)
                    broadcast_data = status.copy()
                    broadcast_data["completed_applets"] = memory_completed_applets
                    await broadcast_status_fn(broadcast_data)
                else:
                    await broadcast_status_fn({
                        "run_id": run_id,
                        "status": "error",
                        "error": "No start node found in flow and status record is invalid",
                        "completed_applets": []
                    })
                return

            context = {
                "input": input_data,
                "results": {},
                "run_id": run_id
            }

            if status and isinstance(status, dict):
                status["input_data"] = input_data

            current_nodes = start_nodes
            visited = set()

            while current_nodes:
                next_nodes = []

                for node_id in current_nodes:
                    if node_id in visited:
                        continue

                    visited.add(node_id)
                    node = nodes_by_id[node_id]

                    if status and isinstance(status, dict):
                        status["current_applet"] = node["type"]
                        status["progress"] += 1

                    if node_id not in memory_completed_applets:
                        memory_completed_applets.append(node_id)

                    if status and isinstance(status, dict):
                        broadcast_data = status.copy()
                        broadcast_data["completed_applets"] = memory_completed_applets
                        await workflow_run_repo.save(status)
                        await broadcast_status_fn(broadcast_data)

                    if node["type"].lower() in ["start", "end"]:
                        if (
                            node["type"].lower() == "start"
                            and "data" in node
                            and "parsedInputData" in node["data"]
                        ):
                            parsed_input = node["data"]["parsedInputData"]
                            if parsed_input and isinstance(parsed_input, dict):
                                context["input"] = parsed_input
                                if status and isinstance(status, dict):
                                    status["input_data"] = parsed_input

                        if node_id in graph:
                            next_nodes.extend(graph[node_id])
                        continue

                    try:
                        applet = await Orchestrator.load_applet(node["type"].lower())

                        message_content = input_data
                        message_metadata = {"node_id": node_id, "run_id": run_id}

                        if "data" in node:
                            if node["type"].lower() == "writer" and "systemPrompt" in node["data"]:
                                message_metadata["system_prompt"] = node["data"]["systemPrompt"]

                            if node["type"].lower() == "artist":
                                if "systemPrompt" in node["data"]:
                                    message_metadata["system_prompt"] = node["data"]["systemPrompt"]
                                if "generator" in node["data"]:
                                    message_metadata["generator"] = node["data"]["generator"]

                        message = AppletMessage(
                            content=message_content,
                            context=context,
                            metadata=message_metadata
                        )

                        response = await applet.on_message(message)

                        context["results"][node_id] = {
                            "type": node["type"],
                            "output": response.content
                        }
                        context.update(response.context)

                        if node_id not in memory_completed_applets:
                            memory_completed_applets.append(node_id)

                        if status and isinstance(status, dict):
                            broadcast_data = status.copy()
                            broadcast_data["completed_applets"] = memory_completed_applets
                            await broadcast_status_fn(broadcast_data)

                        if node_id in graph:
                            next_nodes.extend(graph[node_id])

                            for edge in flow["edges"]:
                                if edge["source"] == node_id:
                                    edge["animated"] = True

                    except Exception as e:
                        logger.error(f"Error executing applet '{node['type']}': {e}")
                        if status and isinstance(status, dict):
                            status["status"] = "error"
                            status["error"] = f"Error in applet '{node['type']}': {str(e)}"
                            status["end_time"] = time.time()
                            await workflow_run_repo.save(status)
                            broadcast_data = status.copy()
                            broadcast_data["completed_applets"] = memory_completed_applets
                            await broadcast_status_fn(broadcast_data)
                        return

                current_nodes = next_nodes

            if status and isinstance(status, dict):
                status["status"] = "success"
                status["end_time"] = time.time()
                status["results"] = context["results"]
                if "input_data" not in status or not status["input_data"]:
                    status["input_data"] = input_data
                await workflow_run_repo.save(status)
                broadcast_data = status.copy()
                broadcast_data["completed_applets"] = memory_completed_applets
                await broadcast_status_fn(broadcast_data)

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            if status and isinstance(status, dict):
                status["status"] = "error"
                status["error"] = f"Workflow execution error: {str(e)}"
                status["end_time"] = time.time()
                await workflow_run_repo.save(status)
                broadcast_data = status.copy()
                broadcast_data["completed_applets"] = memory_completed_applets
                await broadcast_status_fn(broadcast_data)
            else:
                await broadcast_status_fn({
                    "run_id": run_id,
                    "status": "error",
                    "error": f"Workflow execution error: {str(e)}",
                    "completed_applets": memory_completed_applets
                })


# ============================================================
# API Routes (v1)
# ============================================================

v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@v1.get("/applets")
async def list_applets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all registered applets with their metadata (paginated)."""
    result = []

    for applet_type, applet_class in applet_registry.items():
        result.append({
            "type": applet_type,
            **applet_class.get_metadata()
        })

    applets_dir = os.path.join(os.path.dirname(__file__), "..", "applets")
    if os.path.exists(applets_dir):
        for applet_dir in os.listdir(applets_dir):
            if applet_dir.startswith("__") or applet_dir.startswith("."):
                continue
            if applet_dir not in [a["type"] for a in result]:
                try:
                    applet = await Orchestrator.load_applet(applet_dir)
                    result.append({
                        "type": applet_dir,
                        **applet.get_metadata()
                    })
                except Exception as e:
                    logger.warning(f"Failed to load applet '{applet_dir}': {e}")

    return paginate(result, page, page_size)


@v1.post("/flows", status_code=201)
async def create_flow(flow: CreateFlowRequest):
    """Create or update a flow with strict validation."""
    flow_id = flow.id if flow.id else str(uuid.uuid4())

    flow_dict = flow.model_dump()
    flow_dict["id"] = flow_id
    await FlowRepository.save(flow_dict)
    return {"message": "Flow created", "id": flow_id}


@v1.get("/flows")
async def list_flows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all flows with pagination."""
    flows = await FlowRepository.get_all()
    return paginate(flows, page, page_size)


@v1.get("/flows/{flow_id}")
async def get_flow(flow_id: str):
    """Get a flow by ID."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow


@v1.delete("/flows/{flow_id}")
async def delete_flow(flow_id: str):
    """Delete a flow."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    await FlowRepository.delete(flow_id)
    return {"message": "Flow deleted"}


@v1.post("/flows/{flow_id}/run")
async def run_flow(flow_id: str, body: RunFlowRequest):
    """Run a flow with the given input data."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    run_id = await Orchestrator.execute_flow(flow, body.input)
    return {"run_id": run_id}


@v1.get("/runs")
async def list_runs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all workflow runs with pagination."""
    runs = await WorkflowRunRepository.get_all()
    return paginate(runs, page, page_size)


@v1.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get a workflow run by ID."""
    run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@v1.post("/ai/suggest")
async def ai_suggest(body: AISuggestRequest):
    """Generate code suggestions using AI."""
    raise HTTPException(
        status_code=501,
        detail="AI code suggestion is not implemented in the Alpha release. Please check back in a future version."
    )


# Include versioned router
app.include_router(v1)


# ============================================================
# Health Check (unversioned)
# ============================================================

@app.get("/")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SynApps Orchestrator API",
        "version": API_VERSION,
    }


# ============================================================
# WebSocket (versioned, structured protocol with auth)
# ============================================================

@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Authentication phase
    authenticated = WS_AUTH_TOKEN is None  # Auto-auth if no token configured
    session_id: Optional[str] = None

    if not authenticated:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth" and msg.get("token") == WS_AUTH_TOKEN:
                authenticated = True
                session_id = msg.get("session_id")
            else:
                await websocket.send_json(_ws_message("error", {
                    "code": "AUTH_FAILED",
                    "message": "Invalid or missing authentication token",
                }))
                await websocket.close(code=4001, reason="Authentication failed")
                return
        except asyncio.TimeoutError:
            await websocket.send_json(_ws_message("error", {
                "code": "AUTH_TIMEOUT",
                "message": "Authentication timeout - send auth message within 10 seconds",
            }))
            await websocket.close(code=4002, reason="Authentication timeout")
            return
        except (json.JSONDecodeError, Exception):
            await websocket.send_json(_ws_message("error", {
                "code": "AUTH_ERROR",
                "message": "Invalid authentication message format",
            }))
            await websocket.close(code=4003, reason="Invalid auth message")
            return

    # Assign or resume a session
    reconnected = False
    if session_id and session_id in ws_sessions:
        reconnected = True
    else:
        session_id = str(uuid.uuid4())
    ws_sessions[session_id] = "connected"

    # Register client
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected (session={session_id}, reconnected={reconnected}). Total clients: {len(connected_clients)}")

    await websocket.send_json(_ws_message("auth.result", {
        "authenticated": True,
        "session_id": session_id,
        "reconnected": reconnected,
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json(_ws_message("pong"))
                elif msg_type == "subscribe":
                    channel = msg.get("data", {}).get("channel", "")
                    await websocket.send_json(_ws_message("subscribe.ack", {"channel": channel}))
                else:
                    await websocket.send_json(_ws_message("error", {
                        "code": "UNKNOWN_MESSAGE_TYPE",
                        "message": f"Unknown message type: {msg_type}",
                    }))
            except json.JSONDecodeError:
                await websocket.send_json(_ws_message("error", {
                    "code": "INVALID_MESSAGE",
                    "message": "Message must be valid JSON",
                }))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected (session={session_id})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        if session_id:
            ws_sessions[session_id] = "disconnected"
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(connected_clients)}")


# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
