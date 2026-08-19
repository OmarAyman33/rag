# Architecture — Guarded RAG for learning-rag

> System design for the guardrails layer added on top of the original
> learning-rag codebase. **Zero edits to the original files** — everything new
> lives in `guardrails/`, `run_guarded.py`, `ingest_guarded.py`, `app.py`,
> `ui/`, `tests/`, `docs/`.

## 1. Original codebase (untouched)

| File | Role |
|------|------|
| `chroma.py` | Ingestion v2: `.txt`/`.md` → chunks → ChromaDB (`legal_docs`, cosine). **Runs at import time — never imported by new code.** |
| `test_query.py` | Query: embed → retrieve 5 → numbered context → OpenAI `gpt-5.6-luna` via Responses API |
| `less_chaotic.py`, `reranking.py`, `wikipedia.py`, `first_rodeo.py` | Experiments |

## 2. The guarded pipeline

```
question
   │
   ▼
[retrieve]  embed query (embeddinggemma-300m), Chroma cosine, oversample n=20
   │
   ▼
[G1 similarity threshold]   drop chunks with similarity < 0.20
   │
   ▼
[G3 scope gate]             none left → REFUSE, LLM never called
   │
   ▼
[G2 max snippets]           cap at 5 chunks
   │
   ▼
[build context]             numbered "[i] (source, chunk) text"
   │
   ▼
[G4 generate]               Luna (gpt-5.6-luna), temperature 0,
                            knowledge-boundary prompt + citation duty
   │
   ▼
[G5 verify]                 Luna-as-judge: decompose claims → verify vs context
                            → strip unsupported → refuse if nothing survives
   │
   ▼
[P output gate]             redact any personal info before display
   │
   ▼
answer + full diagnostics   (chunks, dropped, verdict, pii_hits)
```

The query CLI (`run_guarded.py`) and the UI backend (`app.py`) both call
`GuardedPipeline.run()`, so CLI and UI behave identically.

## 3. Ingestion pipeline (guarded)

```
input/*.pdf|.txt|.md
   │
   ▼
[read]  pypdf for PDFs, plain read for txt/md
   │
   ▼
[split] RecursiveCharacterTextSplitter 1024/100 (same as original)
   │
   ▼
[P ingestion gate]  block fundamentally personal chunks / redact incidental
   │
   ▼
[embed] SentenceTransformer(embeddinggemma-300m).encode_document
   │
   ▼
[Chroma] collection "legal_docs", cosine, dedupe by source filename
   │
   ▼
report (added / blocked / redacted / clean per file)
```

New vs original: PDF support (original: txt/md only), PII gate, configurable
paths (original hardcoded `/home/omar/spectech/RAG/...`), dedupe + report.

## 4. UI layer

- **Backend** `app.py`: FastAPI.
  - `GET /` → `ui/index.html`
  - `GET /api/status` → corpus + settings
  - `POST /api/ask` `{question, threshold?, max_snippets?}` → guarded answer JSON
  - `POST /api/ingest` (multipart files) → guarded ingestion report
- **Frontend** `ui/index.html`: single-file, vanilla JS.
  - Design: **Warm Terracotta AI Chat Interface** (superdesign, CC0) — cream
    canvas `#FAF6F0`, terracotta accent `#C4552F`, Fraunces/Inter/JetBrains
    Mono, two-pane editorial chat, frameless assistant turns, code blocks,
    model chip "Luna · Guarded", disclaimer, conversation history in
    localStorage, file upload for ingestion.

## 5. Data flow & failure handling

| Failure | Behavior |
|---------|----------|
| Empty corpus / retrieval returns nothing | G3 refusal (no LLM call) |
| All chunks below threshold | G3 refusal |
| Judge parse error | **Fail closed** — refuse rather than serve unverified text |
| Nothing supported after verification | Refusal |
| PII in answer | Redacted before display |
| Presidio missing/broken | Regex recognizers still run |

## 6. Security & privacy posture

- Model calls: same OpenAI client as original (`OPENAI_API_KEY`), model
  `gpt-5.6-luna` (dev/testing, per Sovereign).
- PII never enters the index (ingestion gate) and is redacted from output.
- Read-only agent: retrieval is the only "tool"; no mutation endpoints except
  explicit document ingestion.
- Full run diagnostics returned to the client for audit/logging.

## 7. Running

```bash
pip install -r requirements.txt
python ingest_guarded.py --input-dir input
python run_guarded.py --prompt "your question"
uvicorn app:app --reload --port 8000    # UI
python -m pytest tests/ -v              # tests
```

## 8. Attribution & sources

- Retrieval guardrail semantics (similarity threshold, max context snippets):
  industry-standard RAG practice for grounding and context-window control.
- Verification: RAGAS faithfulness (claim decomposition + NLI).
- PII: Microsoft Presidio + local regex.
- UI design: "Warm Terracotta AI Chat Interface", superdesign prompts (CC0).
- Legal grounding: AI-Agent-RAG-Local research thread — Saudi PDPL (M/19,
  M/148; IR Art 9 anonymisation), GDPR Art 5/6/9, EU AI Act Art 50.