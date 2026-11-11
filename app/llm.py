"""Model factory: the ONLY place in the project that builds the chat model.

Everything comes from .env (MODEL_NAME / API_KEY / BASE_URL / LLM_TEMPERATURE / LLM_EXTRA_BODY),
so any OpenAI-compatible provider (Bailian Qwen, DeepSeek, ...) works without code changes.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.settings import get_settings

LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 2        # transport-level retries inside the client (timeouts, 429, 5xx)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Build the chat model once per process from the current settings."""
    s = get_settings()
    return ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key,
        base_url=s.base_url,
        temperature=s.llm_temperature,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        # Provider-specific switches (e.g. Bailian's enable_thinking) live in .env, not here.
        extra_body=s.llm_extra_body or None,
    )
