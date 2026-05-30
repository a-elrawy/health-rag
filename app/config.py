"""Central configuration. All tunable knobs live here; any can be overridden
via environment variables (or a local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (if present) before reading settings.
load_dotenv()

# Project layout -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
LOG_DIR = BASE_DIR / "logs"
JSONL_LOG_PATH = LOG_DIR / "research_log.jsonl"
SQLITE_LOG_PATH = LOG_DIR / "research_log.db"


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime settings for the RAG pipeline."""

    # --- Chunking (word-based; keeps chunks topical) ----------------------
    chunk_size_words: int = _get_int("CHUNK_SIZE_WORDS", 120)
    chunk_overlap_words: int = _get_int("CHUNK_OVERLAP_WORDS", 30)

    # --- Embeddings (local sentence-transformers model) -------------------
    embedding_model_name: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Retrieval --------------------------------------------------------
    top_k: int = _get_int("TOP_K", 4)

    # --- Evidence sufficiency gate (cosine similarity in [0, 1]) ----------
    # Answer only if the best chunk is clearly relevant AND enough chunks
    # corroborate it.
    min_top_similarity: float = _get_float("MIN_TOP_SIMILARITY", 0.42)
    min_support_similarity: float = _get_float("MIN_SUPPORT_SIMILARITY", 0.32)
    min_supporting_chunks: int = _get_int("MIN_SUPPORTING_CHUNKS", 2)

    # --- Generation (LLM) -------------------------------------------------
    # Answers are generated via the OpenAI Chat Completions API, so any
    # OpenAI-compatible provider works. LLM_PROVIDER picks defaults for the
    # base URL, model, and key env vars; all can be overridden explicitly.
    llm_provider: str = os.environ.get("LLM_PROVIDER", "groq").lower()
    llm_model_name: str = os.environ.get("LLM_MODEL", "")
    llm_api_key: str = field(init=False, default="")
    llm_base_url: str = field(init=False, default="")
    use_llm: bool = field(init=False, default=False)

    # Built-in defaults per provider: (base_url, default_model, [key env vars]).
    _PROVIDER_DEFAULTS = {
        "openai": ("https://api.openai.com/v1", "gpt-4o-mini", ["OPENAI_API_KEY"]),
        "groq": (
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
            ["GROQ_API_KEY", "LLM_API_KEY"],
        ),
        "openrouter": (
            "https://openrouter.ai/api/v1",
            "meta-llama/llama-3.3-70b-instruct",
            ["OPENROUTER_API_KEY", "LLM_API_KEY"],
        ),
    }

    def __post_init__(self) -> None:
        base_url, default_model, key_vars = self._PROVIDER_DEFAULTS.get(
            self.llm_provider, self._PROVIDER_DEFAULTS["openai"]
        )
        self.llm_base_url = os.environ.get("LLM_BASE_URL", base_url)
        if not self.llm_model_name:
            self.llm_model_name = default_model
        # First non-empty key env var wins; LLM_API_KEY is a generic fallback.
        key = ""
        for var in [*key_vars, "LLM_API_KEY"]:
            key = os.environ.get(var, "")
            if key:
                break
        self.llm_api_key = key
        self.use_llm = bool(self.llm_api_key)  # generation requires a key


settings = Settings()
