"""Configuration for the guarded RAG system.

Everything is tunable without touching the original repository code.
Defaults: similarity threshold 0.20, max context snippets 4-6; same stack as
the original repo (embeddinggemma-300m, legal_docs collection, cosine).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Defaults compatible with the original learning-rag stack
DEFAULT_EMBED_MODEL = "google/embeddinggemma-300m"
DEFAULT_CHROMA_PATH = Path("output/chroma_db")
DEFAULT_COLLECTION = "legal_docs"
DEFAULT_LLM_MODEL = "gpt-5.6-luna"  # kept Luna as requested (dev/testing)
DEFAULT_INPUT_DIR = Path("input")


@dataclass
class Settings:
    """All tunables for the guarded pipeline."""

    # --- Retrieval guardrails (G1 / G2) ---
    similarity_threshold: float = 0.20  # G1: drop chunks below this cosine similarity
    max_snippets: int = 5               # G2: max context chunks sent to the LLM

    # --- Model / store ---
    llm_model: str = DEFAULT_LLM_MODEL  # generation + verification judge (Luna)
    embed_model: str = DEFAULT_EMBED_MODEL
    chroma_path: Path = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION
    n_query: int = 20                   # oversample before threshold + cap

    # --- Prompt / generation ---
    temperature: float = 0.0
    max_tokens: int = 2048

    # --- PII (P) ---
    pii_mode: str = "block"             # block | redact | report (see pii.py)
    pii_languages: list = field(default_factory=lambda: ["en", "ar"])
    redact_placeholder: str = "[REDACTED]"

    # --- Paths ---
    input_dir: Path = DEFAULT_INPUT_DIR

    # --- Runtime switches ---
    verbose: bool = False
    run_verification: bool = True       # G5 on/off
    run_output_pii: bool = True         # P output gate on/off

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chroma_path"] = str(self.chroma_path)
        d["input_dir"] = str(self.input_dir)
        return d


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def load_settings(overrides: dict | None = None) -> Settings:
    """Build Settings from env vars + optional dict overrides.

    Env vars: RAG_THRESHOLD, RAG_MAX_SNIPPETS, RAG_LLM_MODEL, RAG_EMBED_MODEL,
    RAG_CHROMA_PATH, RAG_COLLECTION, RAG_PII_MODE, RAG_INPUT_DIR, RAG_VERIFY.
    """
    env = {
        "similarity_threshold": float(os.environ.get("RAG_THRESHOLD", 0.20)),
        "max_snippets": int(os.environ.get("RAG_MAX_SNIPPETS", 5)),
        "llm_model": os.environ.get("RAG_LLM_MODEL", DEFAULT_LLM_MODEL),
        "embed_model": os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        "chroma_path": _env_path("RAG_CHROMA_PATH", DEFAULT_CHROMA_PATH),
        "collection_name": os.environ.get("RAG_COLLECTION", DEFAULT_COLLECTION),
        "pii_mode": os.environ.get("RAG_PII_MODE", "block"),
        "input_dir": _env_path("RAG_INPUT_DIR", DEFAULT_INPUT_DIR),
        "run_verification": os.environ.get("RAG_VERIFY", "1") not in ("0", "false", "False"),
    }
    if overrides:
        env.update(overrides)
    return Settings(**env)
