"""Guarded RAG pipeline - orchestrates retrieval + all guardrails.

Does NOT import the original repository files (chroma.py runs an ingestion loop
at import time, and paths are hardcoded). Instead it re-implements the same
retrieval semantics (SentenceTransformer + ChromaDB cosine + numbered context +
OpenAI Responses API) with configurable paths, and inserts the guardrails.

Flow:
  query -> embed -> retrieve (oversample) -> G1 threshold -> G3 scope gate
        -> G2 max snippets -> build context -> generate (Luna)
        -> G5 verify (Luna judge) -> strip/refuse -> P output filter -> answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .similarity import RetrievedChunk, filter_by_threshold
from .max_snippets import cap_snippets
from .scope import ScopeDecision, decide_scope, DEFAULT_REFUSAL
from .pii import pii_filter_output
from .verification import (
    Verdict,
    VerificationError,
    verify_answer,
    rebuild_from_supported,
)

# Hardened generation prompt (G4): knowledge boundary + authorized abstention
# + citation duty + confidence penalization. Mirrors the original system prompt
# with the grounding contract made explicit.
SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer the user's \
question using ONLY the numbered context passages provided below.

Rules:
- You have no other knowledge. The ONLY source of information is the context.
- If the answer isn't contained in the context, say exactly:
  "I don't know - this is not in your documents." Do not guess or use outside \
knowledge.
- When you use a passage, cite it inline like [1], [2], etc., matching the \
passage numbers below.
- Every factual statement must be supported by a cited passage. A statement \
without a citation is forbidden.
- Be concise. Don't repeat the passages verbatim; synthesize the answer.
- If you are not certain the context supports a claim, omit it.
"""


@dataclass
class GuardedResult:
    """Everything the guarded pipeline produced, for the UI and for logs."""

    question: str
    answer: str = ""
    refused: bool = False
    refusal_reason: str = ""
    context_chunks: list[dict] = field(default_factory=list)
    dropped_by_threshold: int = 0
    citations: list[str] = field(default_factory=list)
    verdict: dict = field(default_factory=dict)
    pii_redacted: int = 0
    pii_hits: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "context_chunks": self.context_chunks,
            "dropped_by_threshold": self.dropped_by_threshold,
            "citations": self.citations,
            "verdict": self.verdict,
            "pii_redacted": self.pii_redacted,
            "pii_hits": self.pii_hits,
            "settings": self.settings,
        }


class GuardedPipeline:
    """Holds lazily-loaded heavy objects (embed model, chroma, openai client)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embed_model = None
        self._collection = None
        self._client = None

    # -- lazy heavy deps -----------------------------------------------------
    def _embed(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self.settings.embed_model)
        return self._embed_model

    def _chroma(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            # get_or_create so querying an empty corpus refuses cleanly (G3)
            # instead of crashing on a missing collection.
            self._collection = client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _llm(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()  # uses OPENAI_API_KEY env, same as original
        return self._client

    # -- core retrieval ------------------------------------------------------
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        collection = self._chroma()

        # Empty corpus -> refuse BEFORE loading the embed model or querying.
        if collection.count() == 0:
            return []

        query_embedding = self._embed().encode_query(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.settings.n_query,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            cid = ids[i] if i < len(ids) else f"chunk_{i}"
            chunks.append(
                RetrievedChunk(id=cid, text=str(doc), distance=float(dist), metadata=dict(meta))
            )
        # Sort by similarity descending (highest relevance first).
        chunks.sort(key=lambda c: c.similarity, reverse=True)
        return chunks

    def generate(self, question: str, context: str) -> str:
        """Generate the answer with Luna (same API as the original test_query)."""
        client = self._llm()
        user_prompt = f"Context passages:\n{context}\n\nQuestion: {question}"
        response = client.responses.create(
            model=self.settings.llm_model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            max_output_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )
        return response.output_text

    # -- main entry ----------------------------------------------------------
    def run(self, question: str) -> GuardedResult:
        s = self.settings
        result = GuardedResult(question=question, settings=s.to_dict())

        # 1. Retrieve (oversample so G1 + G2 have room to work)
        all_chunks = self.retrieve(question)
        if not all_chunks:
            result.refused = True
            result.refusal_reason = "retrieval returned nothing (empty corpus?)"
            result.answer = DEFAULT_REFUSAL
            return result

        # 2. G1 - document similarity threshold
        kept = filter_by_threshold(all_chunks, s.similarity_threshold)
        result.dropped_by_threshold = len(all_chunks) - len(kept)

        # 3. G3 - scope gate: no context -> refuse WITHOUT calling the LLM
        scope = decide_scope(kept, s.similarity_threshold)
        if scope.refused:
            result.refused = True
            result.refusal_reason = scope.reason
            result.answer = DEFAULT_REFUSAL
            return result

        # 4. G2 - max context snippets
        final_chunks = cap_snippets(kept, s.max_snippets)

        # 5. Build the numbered context (same style as original test_query.py)
        context_blocks = []
        for i, chunk in enumerate(final_chunks, start=1):
            src = chunk.metadata.get("source", "unknown")
            idx = chunk.metadata.get("chunk_index", "?")
            context_blocks.append(f"[{i}] (source: {src}, chunk {idx})\n{chunk.text}")
        context = "\n\n".join(context_blocks)

        result.context_chunks = [c.to_dict() for c in final_chunks]
        result.citations = [f"[{i}]" for i in range(1, len(final_chunks) + 1)]

        # 6. Generate (Luna)
        answer = self.generate(question, context)

        # 7. G5 - post-generation verification (fail closed)
        if s.run_verification:
            try:
                verdict = verify_answer(self._llm(), s, answer, context)
                result.verdict = verdict.to_dict()
                if not verdict.fully_grounded:
                    rebuilt = rebuild_from_supported(verdict)
                    if rebuilt is None:
                        result.answer = DEFAULT_REFUSAL
                        result.refused = True
                        result.refusal_reason = "verification: no supported claims remained"
                        return result
                    answer = rebuilt
            except VerificationError:
                # Judge failed -> fail closed: refuse rather than serve unverified text.
                result.answer = DEFAULT_REFUSAL
                result.refused = True
                result.refusal_reason = "verification judge error (fail closed)"
                return result

        # 8. P - output PII gate (redact before display)
        if s.run_output_pii:
            cleaned, hits = pii_filter_output(
                answer, mode="redact", placeholder=s.redact_placeholder
            )
            result.answer = cleaned
            result.pii_redacted = len(hits)
            result.pii_hits = [{"type": h.entity_type, "text": h.text} for h in hits]
        else:
            result.answer = answer

        return result


def run_guarded(question: str, settings: Settings | None = None) -> GuardedResult:
    """One-shot helper: build a pipeline from settings and run a question."""
    settings = settings or load_settings_default()
    return GuardedPipeline(settings).run(question)


def load_settings_default() -> Settings:
    from .config import load_settings

    return load_settings()