import os
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

INPUT_ROOT = Path("/home/omar/spectech/RAG/input")
OUTPUT_ROOT = Path("/home/omar/spectech/RAG/output")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
model = SentenceTransformer("google/embeddinggemma-300m")

def to_chunks(document):
    return text_splitter.split_text(document)

def to_embedding(chunk):
    return model.encode_document(chunk)

# iterate over all the document files in the input root and call the chunking function. 
for document_path in Path.iterdir(Path(INPUT_ROOT)):
    if document_path.is_file():
        # check if it's a .txt or .md file then send to the chunking function
        if document_path.suffix in ['.txt', '.md']:
            with open(document_path, 'r') as f:
                document = f.read()
                chunks = to_chunks(document)

                # writing the chunks with their respective embeddings in a json file
                chunk_file_path = OUTPUT_ROOT / "chunks" / f"{document_path.stem}.chunks.json"
                with open(chunk_file_path, 'w') as chunk_file:
                    data = {}
                    for num,chunk in enumerate(chunks,start=1):
                        embedding_file_path = OUTPUT_ROOT / "chunks" /f"{document_path.stem}_chunk_{num}_embeddings.npy"
                        embedding = to_embedding(chunk)

                        np.save(embedding_file_path, embedding)
                        data[f"chunk_{num}"] = (chunk,embedding_file_path.__str__())
                    json.dump(data, chunk_file)