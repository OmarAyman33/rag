import os
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb

from guardrails.config import load_settings
from guardrails.pii import classify_chunk

settings = load_settings()

INPUT_ROOT = settings.input_dir
OUTPUT_ROOT = settings.chroma_path.parent

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# device="cpu": the GPU is reserved for the Chandra OCR vLLM server.
model = SentenceTransformer(settings.embed_model, device="cpu")

CHROMA_DB_PATH = settings.chroma_path
DB_NAME = settings.collection_name

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_or_create_collection(
    name=DB_NAME,
    metadata={"hnsw:space": "cosine"},  # cosine similarity search
)


def to_chunks(document):
    return text_splitter.split_text(document)

def to_embeddings(chunks):
    return model.encode_document(chunks).tolist()

def already_ingested(source_name):
    # check if a document with this source name is already ingested.
    existing = collection.get(where={"source": source_name}, limit=1)
    return len(existing["ids"]) > 0


def ingest_text(text: str, source_name: str) -> dict:
    """Chunk -> PII gate (P) -> embed -> add to the corpus. Shared by the
    directory-walk script below and the /api/ingest upload endpoint."""
    if already_ingested(source_name):
        return {"source": source_name, "status": "skipped", "reason": "already ingested", "chunks": 0, "blocked": 0, "redacted": 0}

    raw_chunks = to_chunks(text)
    if not raw_chunks:
        return {"source": source_name, "status": "skipped", "reason": "no text extracted", "chunks": 0, "blocked": 0, "redacted": 0}

    # P: gate every chunk for personal information before it is
    # embedded/stored (block | redact | report, per RAG_PII_MODE).
    chunks = []
    blocked = 0
    redacted = 0
    for chunk in raw_chunks:
        classification = classify_chunk(chunk, settings.pii_mode)
        if classification.action == "block":
            blocked += 1
            continue
        if classification.action == "redact":
            redacted += 1
            chunks.append(classification.redacted_text)
        else:
            chunks.append(chunk)

    if not chunks:
        return {
            "source": source_name,
            "status": "blocked",
            "reason": f"all {len(raw_chunks)} chunks blocked by PII gate",
            "chunks": 0,
            "blocked": blocked,
            "redacted": redacted,
        }

    embeddings = to_embeddings(chunks)

    stem = Path(source_name).stem
    ids = [f"{stem}_chunk_{i}" for i in range(1, len(chunks) + 1)]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(1, len(chunks) + 1)]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    return {"source": source_name, "status": "ingested", "chunks": len(chunks), "blocked": blocked, "redacted": redacted}


def _ingest_directory():
    # iterate over all the document files in the input root and ingest them.
    for document_path in Path.iterdir(Path(INPUT_ROOT)):
        if not document_path.is_file() or document_path.suffix not in [".txt", ".md"]:
            continue

        document = document_path.read_text(encoding="utf-8", errors="ignore")
        report = ingest_text(document, document_path.name)

        if report["status"] == "ingested":
            pii_note = f" ({report['blocked']} blocked, {report['redacted']} redacted)" if report["blocked"] or report["redacted"] else ""
            print(f"Ingested {document_path.name}: {report['chunks']} chunks{pii_note}")
        elif report["status"] in ("blocked", "skipped"):
            print(f"Skipped {document_path.name}: {report['reason']}")


if __name__ == "__main__":
    _ingest_directory()