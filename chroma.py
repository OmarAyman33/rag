import os
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb

INPUT_ROOT = Path("/home/omar/spectech/RAG/input")
OUTPUT_ROOT = Path("/home/omar/spectech/RAG/output")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
model = SentenceTransformer("google/embeddinggemma-300m")

CHROMA_DB_PATH = OUTPUT_ROOT / "chroma_db"
DB_NAME = "legal_docs"

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

            document = document_path.read_text()
            chunks = to_chunks(document)
            if not chunks:
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
        
            print(f"Ingested {document_path.name}: {len(chunks)} chunks")