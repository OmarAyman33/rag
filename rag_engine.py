from typing import Iterator

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from guardrails.config import load_settings
from guardrails.similarity import RetrievedChunk, filter_by_threshold
from guardrails.max_snippets import cap_snippets
from guardrails.scope import decide_scope, DEFAULT_REFUSAL
from guardrails.pii import pii_filter_output
from guardrails.verification import verify_answer, rebuild_from_supported, VerificationError

load_dotenv()

settings = load_settings()

CHROMA_DB_PATH = settings.chroma_path
COLLECTION_NAME = settings.collection_name


# Hardened generation prompt (G4): knowledge boundary + authorized abstention
# + mandatory citations + confidence penalization.
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

SPLIT_SYSTEM_PROMPT = """You break a user's question down into simple, atomic \
sub-questions for a document retrieval system.

Rules:
- Use as FEW sub-questions as possible. Only split when the question asks \
about genuinely distinct pieces of information that would need different \
evidence to answer. Do not split out every angle, criterion, or scenario you \
can think of.
- Each sub-question must be self-contained and answerable independently.
- Output ONE sub-question per line, with no numbering, bullets, or extra \
commentary.
- If the question is already simple and atomic, just output it unchanged as \
the only line.
"""

model = SentenceTransformer(settings.embed_model)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)

client = OpenAI()


def run_rag_query(query: str) -> Iterator[dict]:
    """Run the atomic-question-splitting RAG pipeline, yielding structured
    progress/result events suitable for streaming to a client (e.g. over SSE).
    """
    try:
        split_response = client.responses.create(
            model=settings.llm_model,
            instructions=SPLIT_SYSTEM_PROMPT,
            input=query,
        )

        atomic_questions = [
            line.strip().lstrip("-*").strip()
            for line in split_response.output_text.splitlines()
            if line.strip()
        ]
        if not atomic_questions:
            atomic_questions = [query]

        yield {"type": "atomic_questions", "questions": atomic_questions}

        # Retrieve chunks per atomic question, deduping by chunk id while
        # remembering which atomic question(s) surfaced each chunk
        merged_chunks = {}
        chunks_by_question = {q: [] for q in atomic_questions}
        for atomic_question in atomic_questions:
            atomic_embedding = model.encode_query(atomic_question).tolist()
            results = collection.query(
                query_embeddings=[atomic_embedding],
                n_results=settings.n_query,
            )
            for chunk_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if chunk_id not in merged_chunks:
                    merged_chunks[chunk_id] = {"doc": doc, "meta": meta, "dist": dist}
                chunks_by_question[atomic_question].append(chunk_id)

        # Assign each deduped chunk a stable global number in first-seen order
        chunk_numbers = {chunk_id: i for i, chunk_id in enumerate(merged_chunks, start=1)}

        for atomic_question, chunk_ids in chunks_by_question.items():
            yield {
                "type": "retrieval",
                "question": atomic_question,
                "chunks": [
                    {
                        "n": chunk_numbers[chunk_id],
                        "source": merged_chunks[chunk_id]["meta"]["source"],
                        "chunk_index": merged_chunks[chunk_id]["meta"]["chunk_index"],
                        "distance": merged_chunks[chunk_id]["dist"],
                    }
                    for chunk_id in chunk_ids
                ],
            }

        yield {
            "type": "citation_map",
            "citations": {
                str(chunk_numbers[chunk_id]): {
                    "source": chunk["meta"]["source"],
                    "chunk_index": chunk["meta"]["chunk_index"],
                    "text": chunk["doc"],
                    "distance": chunk["dist"],
                }
                for chunk_id, chunk in merged_chunks.items()
            },
        }

        # G1: drop chunks below the similarity threshold, sorted best-first.
        retrieved = [
            RetrievedChunk(id=chunk_id, text=chunk["doc"], distance=chunk["dist"], metadata=chunk["meta"])
            for chunk_id, chunk in merged_chunks.items()
        ]
        retrieved.sort(key=lambda c: c.similarity, reverse=True)
        filtered = filter_by_threshold(retrieved, settings.similarity_threshold)

        # G3: refuse outright (no LLM call) if nothing survives the threshold.
        scope = decide_scope(filtered, settings.similarity_threshold)
        if scope.refused:
            yield {"type": "answer_delta", "text": DEFAULT_REFUSAL}
            yield {"type": "done"}
            return

        # G2: cap how many chunks are sent to the LLM.
        capped = cap_snippets(scope.chunks, settings.max_snippets)

        # Build a clearly-structured, numbered context block instead of one blob of text
        context_blocks = []
        for chunk in capped:
            meta = chunk.metadata
            context_blocks.append(
                f"[{chunk_numbers[chunk.id]}] (source: {meta['source']}, chunk {meta['chunk_index']})\n{chunk.text}"
            )

        context = "\n\n".join(context_blocks)

        user_prompt = f"""Context passages:
{context}

Question: {query}"""

        # Generate the full answer first (not streamed) so it can be verified
        # and PII-redacted before anything reaches the client.
        response = client.responses.create(
            model=settings.llm_model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )
        answer = response.output_text

        # G5: verify every claim in the answer against the retrieved context;
        # fail closed (refuse) if the judge can't be parsed or nothing survives.
        if settings.run_verification:
            try:
                verdict = verify_answer(client, settings, answer, context)
            except VerificationError:
                yield {"type": "answer_delta", "text": DEFAULT_REFUSAL}
                yield {"type": "done"}
                return

            if not verdict.fully_grounded:
                rebuilt = rebuild_from_supported(verdict)
                if rebuilt is None:
                    yield {"type": "answer_delta", "text": DEFAULT_REFUSAL}
                    yield {"type": "done"}
                    return
                answer = rebuilt

        # P: redact any personal information before the answer is shown.
        if settings.run_output_pii:
            answer, _pii_hits = pii_filter_output(answer, settings.pii_mode, settings.redact_placeholder)

        # Stream the final, safe answer word-by-word so the UI's incremental
        # rendering keeps working unchanged.
        words = answer.split(" ")
        for i, word in enumerate(words):
            text = word if i == len(words) - 1 else word + " "
            yield {"type": "answer_delta", "text": text}

        yield {"type": "done"}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
