"""Stage-9 evaluation: run eval/eval_cases.jsonl against the REAL agent and score it.

Run from the project root (uses the network and your model quota):
    python -m eval.run_eval                 # all cases once
    python -m eval.run_eval --repeat 3      # 3 rounds: temperature 0 is not fully deterministic
    python -m eval.run_eval --only ret-01 exc-06
Writes eval/results/<timestamp>.jsonl (per case) and eval/report.md (summary + formulas).

Metrics (each case is judged on its final state, see judge()):
  tool_accuracy    = cases whose tool calls include every expected tool and no forbidden tool
                     / cases that declare expected_tools or forbidden_tools
  task_completion  = cases that did not crash AND whose final answer passes the answer checks
                     / all cases
  degradation_rate = exception-category cases that did not crash, produced a non-empty answer
                     AND pass the answer checks / exception-category cases

Infrastructure noise: a run whose final answer is the "model unavailable" message or is empty
says nothing about the agent, so it is re-run (--infra-retries, default 2). Runs that are still
infra-failed after the retries are COUNTED as failures (no silent exclusion) but labelled [infra].
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import AgentRuntime, build_agent, check_high_risk_tools
from app.resilience import model_failure_message
from app.llm import get_fallback_model
from app.mcp_client import SERVERS_FILE, load_mcp_tools
from app.policy import load_policy
from app.tool_logger import ToolCallLog
from core.settings import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "eval"
CASES_FILE = EVAL_DIR / "eval_cases.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_FILE = EVAL_DIR / "report.md"

# Derived from the agent's own message so the two can never drift apart.
MODEL_DOWN_PREFIX = model_failure_message(Exception()).split("(")[0].strip()


def is_infra_failure(run: dict) -> bool:
    """The model API was unreachable or returned nothing: an environment problem, not an agent one."""
    answer = run["answer"].strip()
    return not run["crashed"] and (not answer or answer.startswith(MODEL_DOWN_PREFIX))


# ---------------------------------------------------------------------------
# Runtimes: one per registry variant, built exactly like the production agent
# ---------------------------------------------------------------------------

def registry_file(variant: str, tmp: Path) -> Path:
    """A temporary copy of config/servers.yaml for one evaluation variant.

    Every variant gets an EMPTY file-server workspace, so files left over from manual demos
    cannot leak into answers (evaluations must not depend on the developer's local state).
    'knowledge_down' additionally breaks Server-A to test degradation.
    """
    data = yaml.safe_load(SERVERS_FILE.read_text(encoding="utf-8"))
    servers = data["servers"]
    if "files" in servers:
        workspace = tmp / f"workspace-{variant}"
        workspace.mkdir(parents=True, exist_ok=True)
        servers["files"]["env"] = {**(servers["files"].get("env") or {}), "WORKSPACE_DIR": str(workspace)}
    if variant == "knowledge_down":
        servers["knowledge"]["args"] = ["-m", "servers.knowledge_server_is_down"]
    elif variant != "default":
        raise ValueError(f"unknown registry variant {variant!r}")
    path = tmp / f"servers-{variant}.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


async def build_runtime(variant: str, tmp: Path, tool_log: ToolCallLog) -> AgentRuntime:
    policy = load_policy()
    report = await load_mcp_tools(registry_file(variant, tmp))
    unavailable = check_high_risk_tools(policy, report)
    saver = InMemorySaver()
    graph = build_agent(report.tools, policy, checkpointer=saver,
                        fallback_model=get_fallback_model(), tool_log=tool_log)
    return AgentRuntime(graph=graph, policy=policy, tool_report=report,
                        unknown_high_risk=unavailable, checkpointer=saver)


# ---------------------------------------------------------------------------
# Running and judging one case
# ---------------------------------------------------------------------------

def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else "" for b in content)


async def run_case(runtime: AgentRuntime, case: dict, thread_id: str) -> dict:
    config = runtime.run_config(thread_id)
    decision = {"type": "approve"} if case["approval"] == "approve" else \
               {"type": "reject", "message": "Rejected by the evaluation policy."}
    started = time.perf_counter()
    crashed, error = False, None
    try:
        for turn in case["turns"]:
            await runtime.graph.ainvoke({"messages": [("user", turn)]}, config)
            while pending := await runtime.pending_approvals(thread_id):
                await runtime.graph.ainvoke(runtime.resume([decision] * len(pending)), config)
    except Exception as exc:  # a crash is a result, not a reason to stop the whole evaluation
        crashed, error = True, f"{type(exc).__name__}: {exc}"[:300]
    elapsed = time.perf_counter() - started

    messages = (await runtime.graph.aget_state(config)).values.get("messages", [])
    tools_called = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    answer = text_of(final.content) if final else ""
    meta = (final.response_metadata or {}) if final else {}
    return {"id": case["id"], "category": case["category"], "crashed": crashed, "error": error,
            "seconds": round(elapsed, 1), "tools_called": tools_called, "answer": answer,
            # diagnostics: which model produced the answer (fallback?) and why it stopped
            "answer_model": meta.get("model_name"), "finish_reason": meta.get("finish_reason")}


def judge(case: dict, run: dict) -> dict:
    """Apply the case's checks to one run. Returns flags plus human-readable reasons."""
    called = set(run["tools_called"])
    answer = run["answer"]
    reasons = []

    missing = [t for t in case["expected_tools"] if t not in called]
    misused = [t for t in case["forbidden_tools"] if t in called]
    tool_applicable = bool(case["expected_tools"] or case["forbidden_tools"])
    tool_ok = not missing and not misused
    if missing:
        reasons.append(f"expected tool not called: {missing}")
    if misused:
        reasons.append(f"forbidden tool called: {misused}")

    lacks_all = [k for k in case["must_contain_all"] if k not in answer]
    lacks_any = bool(case["must_contain_any"]) and not any(k in answer for k in case["must_contain_any"])
    leaked = [k for k in case["must_not_contain"] if k in answer]
    answer_ok = not lacks_all and not lacks_any and not leaked and bool(answer.strip())
    if lacks_all:
        reasons.append(f"answer lacks {lacks_all}")
    if lacks_any:
        reasons.append(f"answer has none of {case['must_contain_any']}")
    if leaked:
        reasons.append(f"answer contains forbidden text {leaked}")
    if run["crashed"]:
        reasons.append(f"crashed: {run['error']}")
    if is_infra_failure(run):
        reasons.insert(0, "[infra] model unavailable or empty answer")

    completed = not run["crashed"] and answer_ok
    return {"tool_applicable": tool_applicable, "tool_ok": tool_ok,
            "completed": completed, "reasons": reasons}


# ---------------------------------------------------------------------------
# Metrics and report
# ---------------------------------------------------------------------------

def ratio(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.1%}" if den else "n/a"


def metrics(rows: list[dict]) -> dict:
    tool_rows = [r for r in rows if r["tool_applicable"]]
    exc_rows = [r for r in rows if r["category"] == "exception"]
    return {
        "tool_accuracy": (sum(r["tool_ok"] for r in tool_rows), len(tool_rows)),
        "task_completion": (sum(r["completed"] for r in rows), len(rows)),
        "degradation_rate": (sum(r["completed"] for r in exc_rows), len(exc_rows)),
    }


def write_report(rows: list[dict], rounds: int, results_file: Path, model: str) -> str:
    m = metrics(rows)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lines = [
        "# Evaluation report", "",
        f"- Run at: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Model: `{model}`  |  Cases: {len(rows) // rounds}  |  Rounds: {rounds}",
        f"- Raw results: `{results_file.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Infra re-runs: {sum(r['infra_retries'] for r in rows)} "
        "(runs repeated because the model API was unavailable or returned an empty answer)", "",
        "## Metrics", "",
        "| Metric | Result | Formula |", "|---|---|---|",
        f"| Tool-call accuracy | {ratio(*m['tool_accuracy'])} | runs whose tool calls include every "
        "expected tool and no forbidden tool / runs that declare expected or forbidden tools |",
        f"| Task completion | {ratio(*m['task_completion'])} | runs that did not crash and whose final "
        "answer passes must_contain_all / must_contain_any / must_not_contain / all runs |",
        f"| Degradation success | {ratio(*m['degradation_rate'])} | exception runs that did not crash and "
        "gave a non-empty answer passing its checks / exception runs |", "",
        "## By category", "",
        "| Category | Runs | Tool accuracy | Task completion | Avg seconds |", "|---|---|---|---|---|",
    ]
    for cat, cat_rows in sorted(by_cat.items()):
        cm = metrics(cat_rows)
        avg = sum(r["seconds"] for r in cat_rows) / len(cat_rows)
        lines.append(f"| {cat} | {len(cat_rows)} | {ratio(*cm['tool_accuracy'])} | "
                     f"{ratio(*cm['task_completion'])} | {avg:.1f} |")
    failures = [r for r in rows if not r["completed"] or (r["tool_applicable"] and not r["tool_ok"])]
    lines += ["", "## Failures", ""]
    lines += [f"- `{r['id']}` (round {r['round']}): {'; '.join(r['reasons'])}" for r in failures] or ["None."]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


async def main(args) -> int:
    cases = [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = RESULTS_DIR / f"{stamp}.jsonl"
    tool_log = ToolCallLog(RESULTS_DIR / f"{stamp}-tool_calls.jsonl")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        runtimes = {}
        for variant in sorted({c["registry"] for c in cases}):
            runtimes[variant] = await build_runtime(variant, Path(tmp), tool_log)
        for rnd in range(1, args.repeat + 1):
            for case in cases:
                thread = f"eval-{stamp}-{rnd}-{case['id']}"
                run = await run_case(runtimes[case["registry"]], case, thread)
                retries = 0
                while is_infra_failure(run) and retries < args.infra_retries:
                    retries += 1
                    print(f"[INFRA] r{rnd} {case['id']:<8} model unavailable / empty answer, re-run {retries}")
                    await asyncio.sleep(5 * retries)
                    run = await run_case(runtimes[case["registry"]], case, f"{thread}-retry{retries}")
                row = {**run, **judge(case, run), "round": rnd, "infra_retries": retries}
                rows.append(row)
                mark = "PASS" if row["completed"] and (row["tool_ok"] or not row["tool_applicable"]) else "FAIL"
                print(f"[{mark}] r{rnd} {case['id']:<8} {row['seconds']:>5.1f}s  tools={row['tools_called']}"
                      + (f"  <- {'; '.join(row['reasons'])}" if row["reasons"] else ""))
                with results_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from core.settings import get_settings
    print("\n" + write_report(rows, args.repeat, results_file, get_settings().model_name))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--infra-retries", type=int, default=2,
                        help="re-runs for a case whose answer shows the model API was unavailable")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
