import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from rag_engine import run_rag_query
from chroma import ingest_text
from chandra_ocr import ocr_pdf

app = FastAPI()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/chat")
async def chat(query: str):
    async def event_generator():
        async for event in iterate_in_threadpool(run_rag_query(query)):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/ingest")
def ingest(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        name = f.filename or "unnamed"
        suffix = Path(name).suffix.lower()
        data = f.file.read()
        try:
            if suffix == ".pdf":
                ocr_result = ocr_pdf(data)
                if ocr_result["error"]:
                    results.append({"source": name, "status": "error", "reason": ocr_result["error"]})
                    continue
                report = ingest_text(ocr_result["text"], name)
                report["pages_ocred"] = ocr_result["pages_ocred"]
                report["pages_total"] = ocr_result["pages_total"]
            elif suffix in (".txt", ".md"):
                report = ingest_text(data.decode("utf-8", errors="ignore"), name)
            else:
                report = {"source": name, "status": "error", "reason": f"unsupported file type {suffix}"}
        except Exception as e:
            report = {"source": name, "status": "error", "reason": str(e)}
        results.append(report)
    return {"results": results}


app.mount("/static", StaticFiles(directory="static"), name="static")
