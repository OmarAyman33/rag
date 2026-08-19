"""G3 - Scope gate / hard refusal.

If no chunk passes the similarity threshold, the agent REFUSES and the LLM is
never called. No context -> no generation, ever. This is the single strongest
anti-hallucination rule: the model cannot answer from memory if we never give
it a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_REFUSAL = (
    "I couldn't find an answer to that in your documents. "
    "This question is not covered by the indexed material."
)


@dataclass
class ScopeDecision:
    """Outcome of the scope gate."""

    allowed: bool
    chunks: list  # list[RetrievedChunk], only set when allowed
    reason: str = ""

    @property
    def refused(self) -> bool:
        return not self.allowed


def decide_scope(chunks: list, threshold: float) -> ScopeDecision:
    """G3: decide whether the question is answerable from the corpus.

    Uses the G1-filtered chunk list. If nothing survives the threshold, we
    refuse without calling the LLM.
    """
    if not chunks:
        return ScopeDecision(
            allowed=False,
            chunks=[],
            reason=f"no chunks above similarity threshold {threshold}",
        )
    return ScopeDecision(allowed=True, chunks=chunks, reason="in scope")
