"""G2 - Max context snippets.

Controls how many text chunks are sent to the LLM. Keeping this value between
4 and 6 avoids context-window overflow and excessive noise that can mislead
the model.
"""

from __future__ import annotations

from .similarity import RetrievedChunk


def cap_snippets(chunks: list[RetrievedChunk], max_snippets: int) -> list[RetrievedChunk]:
    """G2: keep only the top `max_snippets` chunks (already sorted by relevance).

    The input list should already be filtered by G1 and sorted by similarity
    descending. This caps the context window size sent to the LLM.
    """
    if max_snippets <= 0:
        return []
    return chunks[:max_snippets]
