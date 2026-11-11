"""Agent layer: assemble model + MCP tools + policy into a LangGraph ReAct agent.

This file must never name a specific tool. Tools come from app.mcp_client (servers.yaml),
limits and the system prompt from config/policy.yaml, the model from app.llm (.env).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.llm import get_chat_model, get_fallback_model
from app.mcp_client import LoadReport, load_mcp_tools
from app.policy import AgentPolicy, load_policy
from app.resilience import build_resilience_middleware
from app.tool_logger import AttemptCounterMiddleware, ToolCallLog, ToolLoggingMiddleware
from core.settings import get_settings

logger = logging.getLogger(__name__)
APPROVAL_DECISIONS = ["approve", "reject"]


def approval_middleware(policy: AgentPolicy) -> list:
    """Pause before every high-risk tool listed in policy.yaml (names never live in this file)."""
    if not policy.high_risk_tools:
        return []
    return [HumanInTheLoopMiddleware(
        interrupt_on={name: {"allowed_decisions": APPROVAL_DECISIONS} for name in policy.high_risk_tools},
        description_prefix="This tool is marked high-risk and needs human approval",
    )]


def build_agent(
    tools: list[BaseTool],
    policy: AgentPolicy,
    model: BaseChatModel | None = None,
    checkpointer: Any = None,
    fallback_model: BaseChatModel | None = None,
    tool_log: ToolCallLog | None = None,
):
    """Create the ReAct agent graph. Models are injectable so tests can use scripted ones."""
    return create_agent(
        model=model or get_chat_model(),
        tools=tools,
        system_prompt=policy.system_prompt,
        middleware=[
            # Human approval before high-risk tools (needs the checkpointer to pause and resume).
            *approval_middleware(policy),
            # Too many LLM calls in one turn: stop and return what we have.
            ModelCallLimitMiddleware(run_limit=policy.max_model_calls, exit_behavior="end"),
            # Too many tool calls in one turn: refuse further calls, let the model answer.
            ToolCallLimitMiddleware(run_limit=policy.max_tool_calls, exit_behavior="continue"),
            # Outermost tool layer: one log line per call with its FINAL outcome and total time.
            *([ToolLoggingMiddleware(tool_log)] if tool_log else []),
            # Retries, timeouts and model fallback: failures degrade the answer, never crash.
            *build_resilience_middleware(policy, fallback_model),
            # Innermost tool layer: counts real executions (1 + retries) for the log line.
            *([AttemptCounterMiddleware(tool_log)] if tool_log else []),
        ],
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


@dataclass
class AgentRuntime:
    """Everything a front end (CLI, FastAPI) needs: the graph, its policy and the tool report."""

    graph: Any
    policy: AgentPolicy
    tool_report: LoadReport
    unknown_high_risk: list[str] = field(default_factory=list)
    checkpointer: Any = None

    def run_config(self, thread_id: str) -> dict:
        """Per-request config: thread_id selects the conversation memory."""
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.policy.recursion_limit,
        }

    async def pending_approvals(self, thread_id: str) -> list[dict]:
        """Tool calls this conversation is paused on, in the order decisions must be given."""
        state = await self.graph.aget_state(self.run_config(thread_id))
        return [request for interrupt in state.interrupts
                for request in (interrupt.value or {}).get("action_requests", [])]

    @staticmethod
    def resume(decisions: list[dict]) -> Command:
        """Input that continues a paused turn, e.g. [{"type": "approve"}] or
        [{"type": "reject", "message": "why"}]: one decision per pending request."""
        return Command(resume={"decisions": decisions})


def check_high_risk_tools(policy: AgentPolicy, report: LoadReport) -> list[str]:
    """Cross-check the approval list against what actually loaded.

    A name whose server loaded fine but has no such tool, or whose prefix matches no configured
    server, is a typo: refuse to start, because the tool would silently skip approval.
    A name whose server failed to load or is disabled is only a warning: it cannot be called anyway.
    """
    loaded_tools = {tool.name for tool in report.tools}
    configured = set(report.loaded) | set(report.failed) | set(report.disabled)
    typos, unavailable = [], []
    for name in policy.high_risk_tools:
        if name in loaded_tools:
            continue
        server = next((s for s in configured if name.startswith(f"{s}_")), None)
        if server in report.failed or server in report.disabled:
            unavailable.append(name)
        else:
            typos.append(name)
    if typos:
        raise ValueError(
            f"approval.high_risk_tools contains unknown tool names {typos}; "
            f"loaded tools are {sorted(loaded_tools)}"
        )
    if unavailable:
        logger.warning("high-risk tools unavailable (server failed or disabled): %s", unavailable)
    return unavailable


async def create_runtime(checkpointer: Any = None) -> AgentRuntime:
    """Load policy and MCP tools, then build the agent. Call once per process / event loop.

    Pass the previous runtime's checkpointer when reloading, so conversations and paused
    approvals survive the reload.
    """
    policy = load_policy()
    report = await load_mcp_tools()
    if not report.tools:
        raise RuntimeError(f"No MCP tools could be loaded: {report.failed}")
    unavailable = check_high_risk_tools(policy, report)
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    graph = build_agent(report.tools, policy, checkpointer=saver, fallback_model=get_fallback_model(),
                        tool_log=ToolCallLog(get_settings().tool_log_path))
    return AgentRuntime(graph=graph, policy=policy, tool_report=report,
                        unknown_high_risk=unavailable, checkpointer=saver)
