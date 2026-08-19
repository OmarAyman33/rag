from pathlib import Path
from typing import Iterator

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OUTPUT_ROOT = Path("C:/Users/omara/Desktop/RAG/learning-rag/output")
CHROMA_DB_PATH = OUTPUT_ROOT / "chroma_db"
COLLECTION_NAME = "legal_docs"


SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer the user's \
question using ONLY the numbered context passages provided below.

Rules:
- If the answer isn't contained in the context, say so plainly. Do not guess \
or use outside knowledge.
- When you use a passage, cite it inline like [1], [2], etc., matching the \
passage numbers below.
- Be concise. Don't repeat the passages verbatim; synthesize the answer.
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

model = SentenceTransformer("google/embeddinggemma-300m")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)

client = OpenAI()


def run_rag_query(query: str) -> Iterator[dict]:
    """Run the atomic-question-splitting RAG pipeline, yielding structured
    progress/result events suitable for streaming to a client (e.g. over SSE).
    """
    try:
        split_response = client.responses.create(
            model="gpt-5.6-luna",
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
                n_results=5,
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

        # Build a clearly-structured, numbered context block instead of one blob of text
        context_blocks = []
        for chunk_id, chunk in merged_chunks.items():
            meta = chunk["meta"]
            context_blocks.append(
                f"[{chunk_numbers[chunk_id]}] (source: {meta['source']}, chunk {meta['chunk_index']})\n{chunk['doc']}"
            )

        context = "\n\n".join(context_blocks)

        user_prompt = f"""Context passages:
{context}

Question: {query}"""

        stream = client.responses.create(
            model="gpt-5.6-luna",
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                yield {"type": "answer_delta", "text": event.delta}

        yield {"type": "done"}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
