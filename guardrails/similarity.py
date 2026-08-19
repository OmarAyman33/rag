"""G1 - Document similarity threshold.

Filters out low-scoring vector chunks to prevent irrelevant data from
reaching the LLM. The repository stores chunks with hnsw:space=cosine, so
Chroma returns cosine DISTANCE (lower = more similar). We convert to cosine
SIMILARITY and keep only chunks with similarity >= threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its score and metadata."""

    id: str
    text: str
    distance: float
    metadata: dict[str, Any]

    @property
    def similarity(self) -> float:
        return similarity_from_distance(self.distance)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "distance": self.distance,
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
        }


def similarity_from_distance(distance: float) -> float:
    """Cosine distance -> cosine similarity (clamped to [-1, 1])."""
    sim = 1.0 - float(distance)
    return max(-1.0, min(1.0, sim))


def filter_by_threshold(
    chunks: list[RetrievedChunk], threshold: float
) -> list[RetrievedChunk]:
    """G1: keep only chunks whose cosine similarity is >= threshold.

    Chunks below the threshold are dropped BEFORE they reach the LLM.
    """
    return [c for c in chunks if c.similarity >= threshold]
