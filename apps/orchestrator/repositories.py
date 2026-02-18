"""
Repository classes for database operations.

This module provides repository classes that handle database operations for the various models.
"""
from typing import List, Dict, Any, Optional
import logging
import uuid
import time
from apps.orchestrator.models import Flow, FlowNode, FlowEdge, WorkflowRun
from sqlalchemy import delete
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from apps.orchestrator.db import get_db_session

# Configure logging
logger = logging.getLogger("repositories")

class FlowRepository:
    """Async repository for Flow operations."""
    @staticmethod
    async def save(flow_data: Dict[str, Any]) -> Dict[str, Any]:
        flow_id = flow_data.get("id") or str(uuid.uuid4())
        async with get_db_session() as session:
            # Check if flow exists
            result = await session.execute(select(Flow).where(Flow.id == flow_id))
            flow = result.scalars().first()
            if flow:
                # Update existing
                flow.name = flow_data.get("name", flow.name)
                await session.execute(delete(FlowNode).where(FlowNode.flow_id == flow_id))
                await session.execute(delete(FlowEdge).where(FlowEdge.flow_id == flow_id))
            else:
                flow = Flow(id=flow_id, name=flow_data.get("name", "Unnamed Flow"))
                session.add(flow)
                await session.flush()
            # Add nodes
            for node_data in flow_data.get("nodes", []):
                pos = node_data.get("position", {"x": 0, "y": 0})
                node = FlowNode(
                    id=node_data.get("id", str(uuid.uuid4())),
                    flow_id=flow.id,
                    type=node_data.get("type", "unknown"),
                    position_x=pos.get("x", 0),
                    position_y=pos.get("y", 0),
                    data=node_data.get("data", {})
                )
                session.add(node)
            # Add edges
            for edge_data in flow_data.get("edges", []):
                edge = FlowEdge(
                    id=edge_data.get("id", str(uuid.uuid4())),
                    flow_id=flow.id,
                    source=edge_data.get("source", ""),
                    target=edge_data.get("target", ""),
                    animated=edge_data.get("animated", False)
                )
                session.add(edge)
            await session.commit()
            result = await session.execute(
                select(Flow)
                .options(selectinload(Flow.nodes), selectinload(Flow.edges))
                .where(Flow.id == flow.id)
            )
            complete_flow = result.scalars().first()
            return complete_flow.to_dict()
    @staticmethod
    async def get_by_id(flow_id: str) -> Optional[Dict[str, Any]]:
        async with get_db_session() as session:
            result = await session.execute(
                select(Flow)
                .options(selectinload(Flow.nodes), selectinload(Flow.edges))
                .where(Flow.id == flow_id)
            )
            flow = result.scalars().first()
            return flow.to_dict() if flow else None

    @staticmethod
    async def get_all() -> List[Dict[str, Any]]:
        async with get_db_session() as session:
            result = await session.execute(
                select(Flow).options(selectinload(Flow.nodes), selectinload(Flow.edges))
            )
            flows = result.scalars().all()
            return [flow.to_dict() for flow in flows]

    @staticmethod
    async def delete(flow_id: str) -> bool:
        async with get_db_session() as session:
            result = await session.execute(select(Flow).where(Flow.id == flow_id))
            flow = result.scalars().first()
            if not flow:
                return False
            await session.delete(flow)
            await session.commit()
            return True

class WorkflowRunRepository:
    """Async repository for WorkflowRun operations."""
    @staticmethod
    async def save(run_data: Dict[str, Any]) -> Dict[str, Any]:
        run_id = run_data.get("run_id") or str(uuid.uuid4())
        async with get_db_session() as session:
            result = await session.execute(select(WorkflowRun).where(WorkflowRun.id == run_id))
            run = result.scalars().first()
            if run:
                # Update
                for field in ["status", "current_applet", "progress", "total_steps", "end_time", "results", "error", "input_data", "completed_applets"]:
                    if field in run_data:
                        setattr(run, field, run_data[field])
            else:
                run = WorkflowRun(
                    id=run_id,
                    flow_id=run_data.get("flow_id"),
                    status=run_data.get("status", "idle"),
                    current_applet=run_data.get("current_applet"),
                    progress=run_data.get("progress", 0),
                    total_steps=run_data.get("total_steps", 0),
                    start_time=run_data.get("start_time", time.time()),
                    end_time=run_data.get("end_time"),
                    results=run_data.get("results", {}),
                    error=run_data.get("error"),
                    input_data=run_data.get("input_data"),
                    completed_applets=run_data.get("completed_applets", [])
                )
                session.add(run)
            await session.commit()
            return run.to_dict()
    @staticmethod
    async def get_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
        async with get_db_session() as session:
            result = await session.execute(select(WorkflowRun).where(WorkflowRun.id == run_id))
            run = result.scalars().first()
            return run.to_dict() if run else None
    @staticmethod
    async def get_all() -> List[Dict[str, Any]]:
        async with get_db_session() as session:
            result = await session.execute(select(WorkflowRun))
            runs = result.scalars().all()
            return [run.to_dict() for run in runs]

