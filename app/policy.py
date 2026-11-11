"""Load and validate config/policy.yaml (agent limits and system prompt)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from core.settings import PROJECT_ROOT

POLICY_FILE = PROJECT_ROOT / "config" / "policy.yaml"
GRAPH_STEPS_PER_MODEL_CALL = 4   # model node + tools node + middleware hooks, with headroom


@dataclass(frozen=True)
class AgentPolicy:
    max_model_calls: int
    max_tool_calls: int
    recursion_limit: int
    system_prompt: str
    tool_timeout_seconds: float = 60.0
    tool_max_retries: int = 2
    retry_initial_delay_seconds: float = 1.0
    high_risk_tools: tuple[str, ...] = ()


def load_policy(path: Path = POLICY_FILE) -> AgentPolicy:
    """Read policy.yaml; any missing or inconsistent value is a ValueError at startup."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    agent = data.get("agent") or {}
    resilience = data.get("resilience") or {}
    approval = data.get("approval") or {}
    high_risk = approval.get("high_risk_tools") or []
    if not isinstance(high_risk, list) or not all(isinstance(n, str) and n for n in high_risk):
        raise ValueError(f"{path}: approval.high_risk_tools must be a list of tool names")
    try:
        policy = AgentPolicy(
            max_model_calls=int(agent["max_model_calls"]),
            max_tool_calls=int(agent["max_tool_calls"]),
            recursion_limit=int(agent["recursion_limit"]),
            system_prompt=str(agent["system_prompt"]).strip(),
            tool_timeout_seconds=float(resilience["tool_timeout_seconds"]),
            tool_max_retries=int(resilience["tool_max_retries"]),
            retry_initial_delay_seconds=float(resilience["retry_initial_delay_seconds"]),
            high_risk_tools=tuple(high_risk),
        )
    except KeyError as missing:
        raise ValueError(f"{path}: setting {missing.args[0]!r} is required") from None

    if policy.max_model_calls < 1 or policy.max_tool_calls < 1:
        raise ValueError(f"{path}: max_model_calls and max_tool_calls must be >= 1")
    if policy.recursion_limit < GRAPH_STEPS_PER_MODEL_CALL * policy.max_model_calls:
        raise ValueError(
            f"{path}: recursion_limit ({policy.recursion_limit}) must be at least "
            f"{GRAPH_STEPS_PER_MODEL_CALL} x max_model_calls, or it fires before the step limit"
        )
    if policy.tool_timeout_seconds <= 0 or policy.tool_max_retries < 0:
        raise ValueError(f"{path}: tool_timeout_seconds must be > 0 and tool_max_retries >= 0")
    if not policy.system_prompt:
        raise ValueError(f"{path}: agent.system_prompt is empty")
    return policy
