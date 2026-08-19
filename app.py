import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from rag_engine import run_rag_query

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


app.mount("/static", StaticFiles(directory="static"), name="static")
