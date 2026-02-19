"""
Centralized configuration for hodoscope processing.

All default constants and environment loading live here.
Config.from_env() is the single place that performs env-loading side effects.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Default constants (single source of truth)
# ---------------------------------------------------------------------------

DEFAULT_SUMMARIZE_MODEL = "openai/gpt-5.2"
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"
DEFAULT_EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
DEFAULT_EMBED_DIM = None
DEFAULT_MAX_WORKERS = 10
DEFAULT_GROUP_BY = "model"
DEFAULT_FPS_ALPHA = 1.0
DEFAULT_FPS_BETA = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env from project root or current directory."""
    for candidate in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
        if candidate.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate)
            except ImportError:
                pass
            return


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Processing configuration for hodoscope.

    Use ``Config()`` for hardcoded defaults (no env magic).
    Use ``Config.from_env()`` to resolve from .env + env vars + overrides.
    """

    summarize_model: str = DEFAULT_SUMMARIZE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embed_dim: int | None = DEFAULT_EMBED_DIM
    max_workers: int = DEFAULT_MAX_WORKERS
    reasoning_effort: str | None = None
    normalize_embeddings: bool = False
    summarize_prompt: str | None = None
    fps_alpha: float = DEFAULT_FPS_ALPHA
    fps_beta: float = DEFAULT_FPS_BETA

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Resolve from .env + env vars, then apply explicit overrides.

        This is the only place that does env loading.
        None-valued overrides are ignored (so unset CLI args don't clobber env).
        """
        _load_env()

        env_embed_dim = os.environ.get("EMBED_DIM")
        kwargs = {
            "summarize_model": os.environ.get("SUMMARIZE_MODEL") or DEFAULT_SUMMARIZE_MODEL,
            "embedding_model": os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
            "embed_dim": int(env_embed_dim) if env_embed_dim else DEFAULT_EMBED_DIM,
            "max_workers": int(os.environ.get("MAX_WORKERS", DEFAULT_MAX_WORKERS)),
            "reasoning_effort": os.environ.get("REASONING_EFFORT"),
            "normalize_embeddings": os.environ.get("NORMALIZE_EMBEDDINGS", "false").lower() not in ("false", "0", "no"),
            "fps_alpha": float(os.environ.get("FPS_ALPHA", DEFAULT_FPS_ALPHA)),
            "fps_beta": float(os.environ.get("FPS_BETA", DEFAULT_FPS_BETA)),
        }
        # Apply explicit overrides (skip None values so unset CLI args don't clobber env)
        for k, v in overrides.items():
            if v is not None:
                kwargs[k] = v
        return cls(**kwargs)
