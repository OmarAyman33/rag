# PII Detection — Documentation

> The personal-information blocker: detects personal information and **stops it
> before it reaches** the vector store (ingestion), the LLM (query context), and
> the user (output). This is the enforcement arm of the privacy research
> (Saudi PDPL: redact-before-embed; GDPR: minimize at the prompt boundary).

## Policy (Sovereign-approved)

| Situation | Action |
|-----------|--------|
| Chunk is **fundamentally personal** (dense identifiers, or multiple high-severity types together) | **BLOCK** — chunk never enters the index |
| Chunk has **incidental** identifiers (one email in a contract) | **REDACT** — identifiers masked to `[REDACTED]`, chunk still indexed |
| No personal information | **CLEAN** — indexed as-is |
| `--pii-mode report` | never blocks/redacts; only logs |

## Recognizers

Two layers, both local:

1. **Deterministic regex (always on, no model download):**
   - `EMAIL` — `user@example.com`
   - `PHONE` — including Saudi mobile format `05xxxxxxxx`
   - `SAUDI_ID` / `IQAMA` — 10-digit number starting with 1 or 2
   - `CREDIT_CARD` — 13–19 digit numbers
   - `IP_ADDRESS`
2. **Microsoft Presidio (optional enhancer):** if `presidio-analyzer` is
   installed, NER-based detection (names, locations, etc.) is merged in.
   Presidio failures are swallowed — the deterministic layer always runs.

## The "fundamentally personal" heuristic

A chunk is **blocked** when any of:

- ≥ 2 distinct high-severity types appear (`SAUDI_ID`, `PHONE`,
  `CREDIT_CARD`, `EMAIL`)
- ≥ 2 high-severity hits in one chunk
- identifiers cover > 30% of the chunk text

Otherwise the chunk is **redacted** (identifiers masked) and kept.

## Where the gates sit

```
DOCUMENT → [P ingestion gate: block/redact] → chunks → embeddings → Chroma
                                                    ↑
QUESTION → [query PII check: refuse if PII in question?] → retrieve → G1/G3/G2
                                                    ↓
USER  ← [P output gate: redact any PII in answer] ← G5 verification ← LLM
```

- **Ingestion gate** (`ingest_guarded.py`): the gate that matters most — the
  index never contains personal information, so retrieval can never surface it.
- **Output gate** (`pii_filter_output`): defense in depth — even if a source
  chunk slipped through, identifiers are redacted before the user sees them.
- **Query gate**: the question itself is scanned and identifiers redacted
  before anything is logged.

## Privacy law note (why this design)

- Saudi PDPL: storing personal data in an index is "processing"; **anonymised
  data is outside the law** (Implementing Regulation Art 9) → redact-before-embed
  keeps the index out of scope.
- GDPR Art 5: data minimisation — the agent should only see what it needs.
- The ONLY lawful personal-data moments: the user's own data (consent/contract),
  anonymised data, documented legitimate interest (never sensitive), public
  source. The blocker enforces exactly this.

## Tunables

| Setting | Default | Meaning |
|---------|---------|---------|
| `RAG_PII_MODE` / `--pii-mode` | `block` | `block` / `redact` / `report` |
| `redact_placeholder` | `[REDACTED]` | mask text |

## Known gaps

- Arabic NER accuracy of Presidio is untested; the regex recognizers cover
  Saudi IDs / mobile formats regardless of language.
- Names without context (single "Ahmed" alone) are not blocked by regex; only
  Presidio NER would catch them — add to a deny-list if the corpus requires it.