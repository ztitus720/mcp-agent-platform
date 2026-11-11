"""Stage-0 environment check: Python version, .env, LLM, tool calling, embedding.

Run from the project root:
    python -m scripts.check_env
Exit code is 0 only if every check passes.
"""

import sys
import time
from collections.abc import Callable

REQUIRED_PYTHON = (3, 11)
results: list[tuple[str, bool]] = []


def run_check(name: str, fn: Callable[[], str]) -> bool:
    """Run one check, print a single status line, never let it crash the script."""
    start = time.perf_counter()
    try:
        detail, ok = fn(), True
    except Exception as exc:
        detail, ok = f"{type(exc).__name__}: {exc}"[:300], False
    elapsed = time.perf_counter() - start
    print(f"[{'OK  ' if ok else 'FAIL'}] {name:<14} {elapsed:5.2f}s  {detail}")
    results.append((name, ok))
    return ok


def check_python() -> str:
    v = sys.version_info
    if (v.major, v.minor) != REQUIRED_PYTHON:
        raise RuntimeError(f"need Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}, got {v.major}.{v.minor}")
    return f"{v.major}.{v.minor}.{v.micro} at {sys.executable}"


def check_settings() -> str:
    from core.settings import get_settings, mask_secret

    s = get_settings()
    return (f"model={s.model_name} key={mask_secret(s.api_key)} "
            f"embedding={s.embedding_model}/{s.embedding_dim} search={s.search_mode}")


def _chat_model():
    from langchain_openai import ChatOpenAI
    from core.settings import get_settings

    s = get_settings()
    return ChatOpenAI(model=s.model_name, api_key=s.api_key, base_url=s.base_url,
                      temperature=s.llm_temperature, timeout=30, max_retries=1)


def check_llm() -> str:
    reply = _chat_model().invoke("Reply with exactly one word: pong")
    return f"reply={reply.content!r}"


def probe_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"


def check_tool_calling() -> str:
    llm = _chat_model().bind_tools([probe_weather])
    reply = llm.invoke("What is the weather in Xi'an right now? Use the tool.")
    if not reply.tool_calls:
        raise RuntimeError("model answered without calling the tool; choose a model with function calling")
    call = reply.tool_calls[0]
    return f"called {call['name']}({call['args']})"


def check_embedding() -> str:
    from langchain_openai import OpenAIEmbeddings
    from core.settings import get_settings

    s = get_settings()
    if s.embedding_provider != "openai_compatible":
        raise RuntimeError(f"provider {s.embedding_provider!r} is not implemented yet")
    emb = OpenAIEmbeddings(model=s.embedding_model, api_key=s.resolved_embedding_api_key,
                           base_url=s.resolved_embedding_base_url, dimensions=s.embedding_dim,
                           check_embedding_ctx_length=False)
    vector = emb.embed_query("Xi'an is the capital of Shaanxi province.")
    if len(vector) != s.embedding_dim:
        raise RuntimeError(f"expected {s.embedding_dim} dims, got {len(vector)}")
    return f"dim={len(vector)} first3={[round(x, 4) for x in vector[:3]]}"


def main() -> int:
    print("== Stage 0 environment check ==")
    run_check("python", check_python)
    if run_check("settings", check_settings):
        run_check("llm", check_llm)
        run_check("tool calling", check_tool_calling)
        run_check("embedding", check_embedding)
    else:
        print("       skipped network checks: fix .env first")
    failed = [name for name, ok in results if not ok]
    print("-" * 60)
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
