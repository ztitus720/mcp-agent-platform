"""Tool-call audit log: one JSON line per logical tool call, retries included.

Two middleware share one ToolCallLog:
  ToolLoggingMiddleware   outermost tool layer: total duration and the FINAL outcome
  AttemptCounterMiddleware innermost tool layer: counts real executions (1 + retries)
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

RESULT_PREVIEW_CHARS = 500


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class ToolCallLog:
    """Append-only JSONL file; safe to share between concurrent requests in one process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = {}

    def count_attempt(self, call_id: str) -> None:
        with self._lock:
            self._attempts[call_id] = self._attempts.get(call_id, 0) + 1

    def record(self, request, result, started: float) -> None:
        call = request.tool_call
        status = getattr(result, "status", "success") if isinstance(result, ToolMessage) else "success"
        content = _text(result.content) if isinstance(result, ToolMessage) else str(result)
        config = getattr(request.runtime, "config", None) or {}
        with self._lock:
            attempts = self._attempts.pop(call["id"], 0)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "thread_id": (config.get("configurable") or {}).get("thread_id"),
            "tool": call["name"],
            "args": call["args"],
            "ok": status != "error",
            "attempts": attempts,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "result": content[:RESULT_PREVIEW_CHARS],
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class ToolLoggingMiddleware(AgentMiddleware):
    def __init__(self, log: ToolCallLog) -> None:
        super().__init__()
        self.log = log

    async def awrap_tool_call(self, request, handler):
        started = time.perf_counter()
        result = await handler(request)
        self.log.record(request, result, started)
        return result

    def wrap_tool_call(self, request, handler):
        started = time.perf_counter()
        result = handler(request)
        self.log.record(request, result, started)
        return result


class AttemptCounterMiddleware(AgentMiddleware):
    def __init__(self, log: ToolCallLog) -> None:
        super().__init__()
        self.log = log

    async def awrap_tool_call(self, request, handler):
        self.log.count_attempt(request.tool_call["id"])
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        self.log.count_attempt(request.tool_call["id"])
        return handler(request)
