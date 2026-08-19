"""run_guarded.py - guarded query CLI (NEW code, original repo untouched).

Usage:
  python run_guarded.py --prompt "What does the contract say about termination?"
  python run_guarded.py --prompt "..." --threshold 0.30 --max-snippets 4 --verbose

Guardrails applied: G1 similarity threshold, G2 max snippets, G3 scope gate
(refusal without LLM call), G4 knowledge-boundary prompt, G5 post-generation
verification (Luna judge), P output PII redaction.
"""

from __future__ import annotations

import argparse
import json
import sys

from guardrails.config import load_settings
from guardrails.pipeline import GuardedPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the guarded RAG agent a question.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, default="what is python?",
                        help="The query given to the RAG system")
    parser.add_argument("--threshold", type=float, default=None,
                        help="G1: minimum cosine similarity (default 0.20)")
    parser.add_argument("--max-snippets", type=int, default=None,
                        help="G2: max context chunks sent to the LLM")
    parser.add_argument("--no-verify", action="store_true",
                        help="disable G5 post-generation verification")
    parser.add_argument("--no-output-pii", action="store_true",
                        help="disable the P output redaction gate")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the full result as JSON (for the UI/scripts)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.threshold is not None:
        overrides["similarity_threshold"] = args.threshold
    if args.max_snippets is not None:
        overrides["max_snippets"] = args.max_snippets
    if args.no_verify:
        overrides["run_verification"] = False
    if args.no_output_pii:
        overrides["run_output_pii"] = False
    overrides["verbose"] = args.verbose

    settings = load_settings(overrides)
    pipeline = GuardedPipeline(settings)
    result = pipeline.run(args.prompt)

    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"\nQ: {result.question}")
    if settings.verbose:
        print(f"   threshold={settings.similarity_threshold} "
              f"max_snippets={settings.max_snippets} "
              f"dropped_by_threshold={result.dropped_by_threshold}")
        for c in result.context_chunks:
            print(f"   [{c['metadata'].get('source', '?')} chunk "
                  f"{c['metadata'].get('chunk_index', '?')}] sim={c['similarity']:.4f}")
    if result.refused:
        print(f"\n(REFUSED) {result.refusal_reason}\n{result.answer}")
    else:
        print(f"\nA: {result.answer}")
        if result.verdict:
            print(f"\n[verified] faithfulness={result.verdict.get('faithfulness')} "
                  f"({len(result.verdict.get('supported', []))} supported, "
                  f"{len(result.verdict.get('unsupported', []))} stripped)")
        if result.pii_redacted:
            print(f"[pii] {result.pii_redacted} identifier(s) redacted from output")
    return 0


if __name__ == "__main__":
    sys.exit(main())