# RAG

## Setup

```bash
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:

```
OPENAI_API_KEY=sk-...
```

PDF uploads are OCR'd via Chandra OCR against a vLLM server. If you're
running one, point at it in `.env` (defaults shown):

```
VLLM_API_BASE=http://localhost:8001/v1   # Chandra's own default (8000) collides with this app's port
VLLM_API_KEY=EMPTY
VLLM_MODEL_NAME=chandra
```

No vLLM server configured/reachable? PDF uploads just fail cleanly with an
"OCR backend unavailable" message — `.txt`/`.md` uploads work regardless.

## Ingest documents

```bash
python3 chroma.py
```

## Run the app

```bash
python3 -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000
