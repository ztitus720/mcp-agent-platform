"""HTTP front end: FastAPI + Server-Sent Events, with human approval and introspection.

Run from the project root:
    uvicorn app.api:app --port 8000
Endpoints:
    GET  /health         which MCP servers loaded / failed
    GET  /tools          tools the agent can use right now
    POST /admin/reload   re-read servers.yaml + policy.yaml without restarting (X-Admin-Token)
    POST /chat           {"message", "thread_id"?}            -> SSE stream
    POST /chat/approve   {"thread_id", "decisions": [...]}    -> SSE stream (resumes a paused turn)
SSE events: thread, token, tool_call, tool_result, approval_required, done, error.
"""

import asyncio
import json
import logging
import secrets
import uuid
from collections import defaultdict
from collections.abc import AsyncIterable, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field

from app.agent import AgentRuntime, create_runtime
from core.settings import get_settings

logger = logging.getLogger(__name__)
THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)


class Decision(BaseModel):
    type: Literal["approve", "reject"]
    message: str | None = Field(default=None, max_length=500)


class ApproveRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    decisions: list[Decision] = Field(min_length=1)


def sse(event: str, payload: dict) -> ServerSentEvent:
    """JSON payload with readable non-ASCII text (json.dumps never emits raw newlines)."""
    return ServerSentEvent(event=event, raw_data=json.dumps(payload, ensure_ascii=False, default=str))


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else "" for b in content)


async def run_events(runtime: AgentRuntime, thread_id: str, graph_input) -> AsyncIterable[ServerSentEvent]:
    """Drive the graph and translate what happens into SSE events."""
    yield sse("thread", {"thread_id": thread_id})
    seen: set[str] = set()
    try:
        async for mode, chunk in runtime.graph.astream(
            graph_input, runtime.run_config(thread_id), stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                message, _meta = chunk
                if isinstance(message, AIMessageChunk) and (text := _text(message.content)):
                    yield sse("token", {"text": text})
                continue
            for node_output in chunk.values():
                if not isinstance(node_output, dict):
                    continue
                for message in node_output.get("messages", []):
                    if message.id in seen:
                        continue
                    seen.add(message.id)
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            yield sse("tool_call", {"name": call["name"], "args": call["args"]})
                    elif isinstance(message, ToolMessage):
                        yield sse("tool_result", {
                            "name": message.name, "ok": message.status != "error",
                            "content": _text(message.content)[:500]})

        pending = await runtime.pending_approvals(thread_id)
        if pending:
            yield sse("approval_required", {
                "thread_id": thread_id,
                "requests": [{"name": p["name"], "args": p["args"]} for p in pending]})
            return
        state = await runtime.graph.aget_state(runtime.run_config(thread_id))
        last = state.values["messages"][-1]
        yield sse("done", {"thread_id": thread_id, "answer": _text(last.content)})
    except Exception as exc:  # never leave the client hanging on a half-open stream
        logger.exception("chat stream failed")
        yield sse("error", {"type": type(exc).__name__, "message": str(exc)[:300]})


RuntimeFactory = Callable[..., Awaitable[AgentRuntime]]   # called as factory(checkpointer=...)


def create_app(runtime_factory: RuntimeFactory = create_runtime) -> FastAPI:
    """Build the app. Tests inject a factory that returns a runtime with a scripted model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Tools are bound to the event loop they are loaded in: load them in the server's loop.
        app.state.runtime = await runtime_factory()
        app.state.locks = defaultdict(asyncio.Lock)
        app.state.reload_lock = asyncio.Lock()
        yield

    app = FastAPI(title="MCP Agent Platform", lifespan=lifespan)

    def runtime_of(request: Request) -> AgentRuntime:
        return request.app.state.runtime

    async def claim(request: Request, thread_id: str) -> asyncio.Lock:
        """One turn per conversation at a time: a second request gets 409 instead of corrupting state."""
        lock: asyncio.Lock = request.app.state.locks[thread_id]
        if lock.locked():
            raise HTTPException(409, f"Conversation {thread_id} is busy")
        await lock.acquire()
        return lock

    # Validation runs in dependencies: they finish BEFORE the stream starts, so errors become
    # real HTTP status codes. (An exception raised inside a streaming body cannot change the status.)

    async def chat_ready(body: ChatRequest, request: Request) -> tuple[AgentRuntime, str, asyncio.Lock]:
        rt = runtime_of(request)
        thread_id = body.thread_id or uuid.uuid4().hex[:12]
        if await rt.pending_approvals(thread_id):
            raise HTTPException(409, "This conversation is waiting for approval: call /chat/approve")
        return rt, thread_id, await claim(request, thread_id)

    async def approve_ready(body: ApproveRequest, request: Request) -> tuple[AgentRuntime, list[dict], asyncio.Lock]:
        rt = runtime_of(request)
        pending = await rt.pending_approvals(body.thread_id)
        if not pending:
            raise HTTPException(409, "Nothing is waiting for approval in this conversation")
        if len(body.decisions) != len(pending):
            raise HTTPException(400, f"Expected {len(pending)} decisions, got {len(body.decisions)}")
        decisions = [d.model_dump(exclude_none=True) for d in body.decisions]
        return rt, decisions, await claim(request, body.thread_id)

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        """Admin endpoints are OFF unless ADMIN_TOKEN is set in .env, and then need that token."""
        expected = get_settings().admin_token
        if expected is None:
            raise HTTPException(403, "Admin endpoints are disabled: set ADMIN_TOKEN in .env")
        if not x_admin_token or not secrets.compare_digest(x_admin_token, expected.get_secret_value()):
            raise HTTPException(401, "Missing or wrong X-Admin-Token")

    @app.post("/admin/reload", dependencies=[Depends(require_admin)])
    async def reload(request: Request) -> dict:
        """Rebuild tools + policy from the config files. Running turns finish on the old runtime;
        new requests use the new one. Conversations and paused approvals are kept (same checkpointer).
        If the new configuration is broken, the old runtime stays in service."""
        async with request.app.state.reload_lock:
            old: AgentRuntime = request.app.state.runtime
            try:
                new = await runtime_factory(checkpointer=old.checkpointer)
            except Exception as exc:
                logger.exception("reload failed, keeping the current runtime")
                raise HTTPException(422, f"Reload failed, nothing changed: {type(exc).__name__}: {exc}") from None
            request.app.state.runtime = new
        before = {t.name for t in old.tool_report.tools}
        after = {t.name for t in new.tool_report.tools}
        return {
            "added_tools": sorted(after - before),
            "removed_tools": sorted(before - after),
            "servers_failed": new.tool_report.failed,
            "high_risk_tools": list(new.policy.high_risk_tools),
        }

    @app.get("/health")
    async def health(request: Request) -> dict:
        rt = runtime_of(request)
        return {
            "status": "degraded" if rt.tool_report.failed else "ok",
            "servers_loaded": rt.tool_report.loaded,
            "servers_failed": rt.tool_report.failed,
            "high_risk_tools": list(rt.policy.high_risk_tools),
            "unknown_high_risk_tools": rt.unknown_high_risk,
        }

    @app.get("/tools")
    async def tools(request: Request) -> list[dict]:
        return [{"name": t.name, "description": (t.description or "").strip().splitlines()[0]}
                for t in runtime_of(request).tool_report.tools]

    @app.post("/chat", response_class=EventSourceResponse)
    async def chat(body: ChatRequest, ready=Depends(chat_ready)) -> AsyncIterable[ServerSentEvent]:
        rt, thread_id, lock = ready
        try:
            async for event in run_events(rt, thread_id, {"messages": [("user", body.message)]}):
                yield event
        finally:
            lock.release()

    @app.post("/chat/approve", response_class=EventSourceResponse)
    async def approve(body: ApproveRequest, ready=Depends(approve_ready)) -> AsyncIterable[ServerSentEvent]:
        rt, decisions, lock = ready
        try:
            async for event in run_events(rt, body.thread_id, rt.resume(decisions)):
                yield event
        finally:
            lock.release()

    return app


app = create_app()
