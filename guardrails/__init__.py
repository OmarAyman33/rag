"""Guarded RAG - guardrail package for learning-rag.

All guardrails live OUTSIDE the original repository code (zero edits to the
original files). This package adds:

  G1  Document similarity threshold   (retrieval relevance gate)
  G2  Max context snippets            (context window cap)
  G3  Scope gate / hard refusal       (no context -> no generation)
  G4  Knowledge-boundary prompt       (citations + explicit refusal sentence)
  G5  Post-generation verification    (claim decomposition + NLI-style judge)
  P   PII blocker                     (ingestion + output + query gates)

Everything is deterministic code except the LLM generation/judge calls.
"""

from .config import Settings, load_settings
from .similarity import filter_by_threshold, similarity_from_distance
from .max_snippets import cap_snippets
from .scope import ScopeDecision, decide_scope
from .pii import (
    PIIHit,
    PIIMode,
    PIIClassification,
    classify_chunk,
    detect_pii,
    redact_text,
    pii_filter_output,
)
from .verification import verify_answer, Verdict
from .pipeline import run_guarded, GuardedResult

__all__ = [
    "Settings",
    "load_settings",
    "filter_by_threshold",
    "similarity_from_distance",
    "cap_snippets",
    "ScopeDecision",
    "decide_scope",
    "PIIHit",
    "PIIMode",
    "PIIClassification",
    "classify_chunk",
    "detect_pii",
    "redact_text",
    "pii_filter_output",
    "verify_answer",
    "Verdict",
    "run_guarded",
    "GuardedResult",
]
