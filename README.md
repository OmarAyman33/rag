# learning-rag

A local, grounded, retrieval-augmented question-answering agent over your own
documents (PDF / txt / md).

**Core promise:** the agent answers **only from your documents** — never from
the model's internal knowledge — via deterministic guardrails, a personal
information blocker, and post-generation verification.

> This repo contains the original learning-rag code plus a **guardrails layer
> that never edits the original files**. Everything added is new code with its
> own documentation.

## Quick start

```bash
pip install -r requirements.txt

# 1. Ingest documents (PDFs supported; personal info is blocked/redacted)
python ingest_guarded.py --input-dir input

# 2. Ask from the CLI
python run_guarded.py --prompt "What does the contract say about termination?"

# 3. Or use the UI (Warm Terracotta chat interface)
uvicorn app:app --reload --port 8000    # open http://127.0.0.1:8000
```

Requires an OpenAI-compatible key for generation (`OPENAI_API_KEY`), same as
the original `test_query.py`. Default model: `gpt-5.6-luna` (dev/testing).

## Guardrails

| # | Guardrail | Default |
|---|-----------|---------|
| G1 | Document similarity threshold | `0.20` |
| G2 | Max context snippets | `5` |
| G3 | Scope gate — no context → refuse, LLM never called | on |
| G4 | Knowledge-boundary prompt + citations + refusal | on |
| G5 | Post-generation verification (claims vs context, Luna judge) | on |
| P | PII blocker: ingestion (block/redact) + output (redact) | `block` |

Docs: [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) ·
[`docs/PII-DETECTION.md`](docs/PII-DETECTION.md) ·
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Layout

```
chroma.py, test_query.py, ...   original code (untouched)
guardrails/                     NEW guardrail modules
run_guarded.py                  NEW guarded query CLI
ingest_guarded.py               NEW guarded ingestion CLI (PDF support)
app.py                          NEW FastAPI backend for the UI
ui/index.html                   NEW chat UI (single file)
tests/test_guardrails.py        NEW test suite (no LLM needed)
docs/                           NEW documentation
```

## Testing

```bash
python -m pytest tests/ -v
```

## Notes

- Personal data policy (Sovereign-approved): fundamentally personal chunks are
  blocked before embedding; incidental identifiers are redacted; output is
  always redacted. Rationale: PDPL IR Art 9 (anonymised data outside the law).
- The agent is read-only: retrieval is its only capability.
- Residual risk is measured, not assumed: every run returns its verification
  verdict for logging; monitor faithfulness over time.