"""app.py - FastAPI backend + static UI for the guarded RAG agent.

Endpoints:
  GET  /               -> serves the chat UI (ui/index.html)
  GET  /api/status     -> corpus + settings summary
  POST /api/ask        -> {question, threshold?, max_snippets?} -> guarded answer
  POST /api/ingest     -> multipart file upload -> guarded ingestion report

Run:  uvicorn app:app --reload --port 8000
UI:   http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from guardrails.config import load_settings
from guardrails.pipeline import GuardedPipeline
from ingest_guarded import ingest, SUPPORTED_SUFFIXES

app = FastAPI(title="learning-rag guarded", version="0.1.0")

_settings = load_settings()
_pipeline = GuardedPipeline(_settings)

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    threshold: float | None = None
    max_snippets: int | None = None


class IngestResponse(BaseModel):
    ok: bool
    report: dict


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/status")
def status() -> JSONResponse:
    try:
        collection = _pipeline._chroma()
        count = collection.count()
        sample = collection.get(limit=10, include=["metadatas"])
        sources = sorted({m.get("source", "?") for m in sample.get("metadatas", [])})
    except Exception as exc:  # collection may not exist yet
        count = 0
        sources = []
        last_error = str(exc)
    else:
        last_error = None

    return JSONResponse(
        {
            "ok": True,
            "collection": _settings.collection_name,
            "chunk_count": count,
            "sample_sources": sources,
            "embed_model": _settings.embed_model,
            "llm_model": _settings.llm_model,
            "similarity_threshold": _settings.similarity_threshold,
            "max_snippets": _settings.max_snippets,
            "pii_mode": _settings.pii_mode,
            "chroma_path": str(_settings.chroma_path),
            "last_error": last_error,
        }
    )


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    overrides = {}
    if req.threshold is not None:
        overrides["similarity_threshold"] = req.threshold
    if req.max_snippets is not None:
        overrides["max_snippets"] = req.max_snippets

    if overrides:
        settings = load_settings(overrides)
        pipeline = GuardedPipeline(settings)
    else:
        settings = _settings
        pipeline = _pipeline

    result = pipeline.run(req.question)
    return JSONResponse(result.to_dict())


@app.post("/api/ingest")
async def upload_ingest(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Accept one or more uploaded documents, save to a temp input dir, ingest."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    saved: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="rag_ingest_") as tmp:
        tmp_dir = Path(tmp)
        for up in files:
            suffix = Path(up.filename or "doc.txt").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                return JSONResponse(
                    {"ok": False, "report": {"error": f"unsupported type {suffix}"}},
                    status_code=400,
                )
            dest = tmp_dir / Path(up.filename or "doc.txt").name
            dest.write_bytes(await up.read())
            saved.append(dest)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        report = ingest(tmp_dir, _settings, splitter, files=saved)

    return JSONResponse({"ok": True, "report": report})


# Mount static assets if the ui folder has more than index.html.
if (UI_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(UI_DIR / "assets")), name="assets")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()