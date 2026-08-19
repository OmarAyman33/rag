import argparse

from rag_engine import run_rag_query

parser = argparse.ArgumentParser(description="An example script handling launch parameters.")
parser.add_argument("--prompt", type=str, default="what is python?", help="The query given to the RAG system")
args = parser.parse_args()

printed_questions = False
printed_retrieval_header = False
printed_answer_header = False

for event in run_rag_query(args.prompt):
    if event["type"] == "atomic_questions":
        print("Atomic questions:")
        for q in event["questions"]:
            print(f"- {q}")
        print()
    elif event["type"] == "retrieval":
        if not printed_retrieval_header:
            print("Retrieved chunks by atomic question:")
            printed_retrieval_header = True
        print(f'\n"{event["question"]}"')
        for chunk in event["chunks"]:
            print(
                f"  [{chunk['n']}] {chunk['source']} chunk {chunk['chunk_index']} "
                f"(dist={chunk['distance']:.4f})"
            )
    elif event["type"] == "citation_map":
        print()
    elif event["type"] == "answer_delta":
        if not printed_answer_header:
            print("\n\n\nFinal Model Response:\n")
            printed_answer_header = True
        print(event["text"], end="", flush=True)
    elif event["type"] == "error":
        print(f"\nError: {event['message']}")
    elif event["type"] == "done":
        print()
