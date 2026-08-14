from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer,CrossEncoder
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

RETRIEVE_K = 20
RERANK_N = 5


SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer the user's \
question using ONLY the numbered context passages provided below.
 
Rules:
- If the answer isn't contained in the context, say so plainly. Do not guess \
or use outside knowledge.
- When you use a passage, cite it inline like [1], [2], etc., matching the \
passage numbers below.
- Be concise. Don't repeat the passages verbatim; synthesize the answer.
"""

embed_model = SentenceTransformer("google/embeddinggemma-300m")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)

query = args.prompt
query_embedding = embed_model.encode_query(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=RETRIEVE_K,
)
candidates = list(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
))
 
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# rerank the 20 candidates
pairs = [(query, doc) for doc, _meta, _dist in candidates]
rerank_scores = rerank_model.predict(pairs)

# pick the top 5 candidates after reranking
reranked = sorted(
    zip(candidates, rerank_scores),
    key=lambda x: x[1],
    reverse=True,  # higher cross-encoder score = more relevant
)[:RERANK_N]


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