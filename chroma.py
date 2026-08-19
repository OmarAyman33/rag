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
model = SentenceTransformer(settings.embed_model)

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

def already_ingested(document_path):
    # check if the file is already ingested before ingesting it again.
    existing = collection.get(where={"source": document_path.name}, limit=1)
    return len(existing["ids"]) > 0


# iterate over all the document files in the input root and call the chunking function. 
for document_path in Path.iterdir(Path(INPUT_ROOT)):
    if document_path.is_file():
        # check if it's a .txt or .md file then send to the chunking function
        if document_path.suffix in ['.txt', '.md']:
            if already_ingested(document_path):
                continue

            document = document_path.read_text(encoding="utf-8", errors="ignore")
            raw_chunks = to_chunks(document)
            if not raw_chunks:
                continue

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
                print(f"Skipped {document_path.name}: all {len(raw_chunks)} chunks blocked by PII gate")
                continue

            embeddings = to_embeddings(chunks)

            ids = [f"{document_path.stem}_chunk_{i}" for i in range(1, len(chunks) + 1)]
            metadatas = [
                {"source": document_path.name, "chunk_index": i}
                for i in range(1, len(chunks) + 1)
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            pii_note = f" ({blocked} blocked, {redacted} redacted)" if blocked or redacted else ""
            print(f"Ingested {document_path.name}: {len(chunks)} chunks{pii_note}")