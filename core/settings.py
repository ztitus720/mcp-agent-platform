"""Central configuration, loaded once from the project-root .env file.

Shared by app/ (agent, API) and servers/ (MCP server processes).
Nothing in this module knows about any specific model vendor.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLACEHOLDER_KEY = "your-api-key-here"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,      # "KEY=" in .env counts as "not set"
        extra="ignore",             # unknown keys in .env are not an error
        validate_default=True,      # run validators on default values too
        protected_namespaces=(),    # allow a field called model_name
    )

    # ----- LLM (any OpenAI-compatible endpoint) -----
    model_name: str
    api_key: SecretStr
    base_url: str
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ----- Embedding -----
    embedding_provider: Literal["openai_compatible", "local"] = "openai_compatible"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None

    # ----- Web search -----
    tavily_api_key: SecretStr | None = None

    # ----- Local paths -----
    chroma_persist_dir: Path = Path("chroma_db")
    chroma_collection: str = "knowledge_base"
    tool_log_path: Path = Path("logs/tool_calls.jsonl")

    @field_validator("base_url", "embedding_base_url")
    @classmethod
    def _check_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return v.rstrip("/") if v else v

    @field_validator("chroma_persist_dir", "tool_log_path")
    @classmethod
    def _anchor_to_project_root(cls, v: Path) -> Path:
        # MCP servers run as child processes whose working directory may differ,
        # so relative paths are always resolved against the project root.
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @model_validator(mode="after")
    def _reject_placeholder_key(self) -> "Settings":
        if self.api_key.get_secret_value().strip() in ("", _PLACEHOLDER_KEY):
            raise ValueError("API_KEY in .env is still the placeholder; put your real key there.")
        return self

    # ----- Derived values: embedding falls back to the LLM endpoint -----
    @property
    def resolved_embedding_api_key(self) -> SecretStr:
        return self.embedding_api_key if self.embedding_api_key is not None else self.api_key

    @property
    def resolved_embedding_base_url(self) -> str:
        return self.embedding_base_url if self.embedding_base_url is not None else self.base_url

    @property
    def search_mode(self) -> Literal["tavily", "mock"]:
        return "tavily" if self.tavily_api_key is not None else "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process; later calls reuse the same object."""
    return Settings()


def mask_secret(secret: SecretStr | None) -> str:
    """Show just enough of a secret to tell keys apart, never the whole value."""
    if secret is None:
        return "<not set>"
    value = secret.get_secret_value()
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-4:]}"