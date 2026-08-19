# Guardrails — Documentation

> What we added, why, and how. All code is NEW — the original repository files
> (`chroma.py`, `test_query.py`, etc.) are untouched.

## The problem

A RAG agent can answer from the model's pre-training instead of the documents.
The retrieval stack solves part of this with two settings (a similarity
threshold and a context cap); we implement those two exactly, then add the
layers that close the remaining gap (post-generation verification + PII
blocking).

## The guardrails (in pipeline order)

| # | Guardrail | What it does |
|---|-----------|--------------|
| G1 | **Document similarity threshold** | Drops chunks whose cosine similarity is below `similarity_threshold` (default `0.20`), so low-scoring chunks never reach the LLM. Configured via `--threshold` or `RAG_THRESHOLD`. |
| G3 | **Scope gate / hard refusal** | If nothing passes G1, the agent refuses and the LLM is **never called**. No context → no generation. |
| G2 | **Max context snippets** | Caps the number of chunks sent to the LLM at `max_snippets` (default `5`), avoiding context overflow and noise. Configured via `--max-snippets` or `RAG_MAX_SNIPPETS`. |
| G4 | **Knowledge-boundary prompt** | System prompt states: only source is the context; exact refusal sentence; citation duty; no unsupported claims. |
| G5 | **Post-generation verification** | Luna (as judge) decomposes the answer into atomic claims and checks each against the context (RAGAS faithfulness procedure). Unsupported claims are stripped; if nothing survives → refusal. Fail-closed on judge errors. |
| P | **PII blocker** | See `PII-DETECTION.md`. Gates at ingestion, query, and output. |

## Files added (zero edits to original code)

```
guardrails/__init__.py      package exports
guardrails/config.py        Settings: thresholds, model, paths, PII mode
guardrails/similarity.py    G1: cosine distance -> similarity, threshold filter
guardrails/max_snippets.py  G2: context cap
guardrails/scope.py         G3: refusal decision
guardrails/pii.py           P: detection / classification / redaction
guardrails/verification.py  G5: claim extraction + verification (Luna judge)
guardrails/pipeline.py      orchestration: retrieve -> G1 -> G3 -> G2 -> gen -> G5 -> P
run_guarded.py              CLI: ask a guarded question
ingest_guarded.py           CLI: guarded ingestion (PDF/txt/md)
app.py                      FastAPI backend for the UI
ui/index.html               Warm Terracotta chat UI (single file)
tests/test_guardrails.py    pytest suite (no LLM needed)
docs/ARCHITECTURE.md        system design
docs/GUARDRAILS.md          this file
docs/PII-DETECTION.md       PII blocker design
```

## Why "never hallucinate" (honest engineering)

You cannot erase a model's pre-training, but you can make it structurally
impossible for an ungrounded answer to reach the user:

1. The LLM receives **only** retrieved chunks + the question.
2. If no chunk passes the similarity threshold, the LLM is **not called** (G3).
3. The prompt **requires citations** and **authorizes abstention** (G4).
4. After generation, every claim is **verified against the chunks**; unsupported
   claims are stripped, and if nothing survives the answer is a refusal (G5).
5. Every run is logged with its verification verdict, so a failure is
   attributable to a specific claim and fixable.

Residual risk: the judge can miss nuance (especially in Arabic — see Gaps in
`../work` research). Mitigation: keep thresholds, monitor faithfulness nightly.

## How to use

```bash
# Ask (defaults: threshold 0.20, max snippets 5, Luna model)
python run_guarded.py --prompt "What does the contract say about termination?"

# Tune
python run_guarded.py --prompt "..." --threshold 0.30 --max-snippets 4 --verbose

# Ingest documents (PDF/txt/md) through the PII gate
python ingest_guarded.py --input-dir input --pii-mode block

# Run the UI
uvicorn app:app --reload --port 8000    # then open http://127.0.0.1:8000
```

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `RAG_THRESHOLD` | `0.20` | G1 similarity threshold |
| `RAG_MAX_SNIPPETS` | `5` | G2 context cap |
| `RAG_LLM_MODEL` | `gpt-5.6-luna` | generation + judge model |
| `RAG_EMBED_MODEL` | `google/embeddinggemma-300m` | embedding model |
| `RAG_CHROMA_PATH` | `output/chroma_db` | vector store path |
| `RAG_COLLECTION` | `legal_docs` | collection name |
| `RAG_PII_MODE` | `block` | `block` / `redact` / `report` |
| `RAG_INPUT_DIR` | `input` | ingestion scan folder |
| `RAG_VERIFY` | `1` | disable G5 with `0` |

## Testing

```bash
python -m pytest tests/ -v
```

The suite covers G1, G2, G3, P, and G5 helpers with a fake judge — it does not
call the LLM or require a populated vector store.

## Attribution

- G5 mechanism: RAGAS faithfulness (claim decomposition + NLI verification),
  docs.ragas.io.
- PII: Microsoft Presidio (optional enhancer) + local deterministic recognizers.
- UI: "Warm Terracotta AI Chat Interface" (superdesign, CC0-1.0) design system.