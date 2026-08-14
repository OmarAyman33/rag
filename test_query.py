from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="An example script handling launch parameters.")

# 2. Load the query
parser.add_argument("--prompt", type=str, default="what is python?", help="The query given to the RAG system")
args = parser.parse_args()



OUTPUT_ROOT = Path("/home/omar/spectech/RAG/output")
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

model = SentenceTransformer("google/embeddinggemma-300m")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)

query = args.prompt
query_embedding = model.encode_query(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
)

# Build a clearly-structured, numbered context block instead of one blob of text
context_blocks = []
for i, (doc, meta, dist) in enumerate(
    zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
    start=1,
):
    print(f"[{i}] {meta['source']} chunk {meta['chunk_index']} (dist={dist:.4f})")
    print(doc, "...\n")
    context_blocks.append(
        f"[{i}] (source: {meta['source']}, chunk {meta['chunk_index']})\n{doc}"
    )
 
context = "\n\n".join(context_blocks)
 
user_prompt = f"""Context passages:
{context}
 
Question: {query}"""
 
client = OpenAI()
 
response = client.responses.create(
    model="gpt-5.6-luna",
    instructions=SYSTEM_PROMPT,
    input=user_prompt,
)
 
print("\n\n\nFinal Model Response:\n")
print(response.output_text)