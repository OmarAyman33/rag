"""Chandra OCR over an OpenAI-compatible vLLM endpoint.

Adapted from ../spectech-internship/pipeline_core/ocr/chandra_vllm.py and
backend.py's reachability-first pattern, simplified for this project: vLLM
only (no HF fallback), and only plain markdown text is returned (no layout
blocks/bboxes, since nothing here needs the UI bounding-box explorer).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from chandra.model.schema import BatchInputItem
from chandra.model.vllm import generate_vllm
from chandra.output import parse_markdown
from chandra.settings import settings

MAX_PDF_PAGES = 15
RENDER_ZOOM = 192 / 72  # matches chandra's own IMAGE_DPI default

# chandra.settings.MAX_OUTPUT_TOKENS (12384) assumes the served model was
# launched with a correspondingly large --max-model-len; if the real server's
# context window is smaller, every request fails and generate_vllm retries
# MAX_VLLM_RETRIES (6) times for a problem retrying can't fix. Override via
# OCR_MAX_OUTPUT_TOKENS if the server's window allows/requires something else.
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def _max_output_tokens() -> int:
    try:
        return max(1, int(os.environ.get("OCR_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)))
    except ValueError:
        return DEFAULT_MAX_OUTPUT_TOKENS


def health_check(timeout: float = 3.0) -> tuple[bool, str | None]:
    """Best-effort ping of the vLLM server's model list endpoint, so an
    unreachable server fails fast instead of burning through generate_vllm's
    own retry loop."""
    from openai import OpenAI

    try:
        client = OpenAI(api_key=settings.VLLM_API_KEY, base_url=settings.VLLM_API_BASE, max_retries=0, timeout=timeout)
        client.models.list()
        return True, None
    except Exception as e:
        return False, str(e)


def _pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES):
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    n_pages = min(total_pages, max_pages)

    matrix = pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)
    images = []
    for i in range(n_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix)
        mode = "RGBA" if pix.n == 4 else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        images.append(img.convert("RGB") if mode == "RGBA" else img)
    doc.close()

    return images, total_pages


def ocr_pdf(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES) -> dict:
    """OCR every page (up to max_pages) of a PDF and return the combined
    markdown text, ready to hand to chroma.ingest_text().

    Returns {"text": str, "pages_ocred": int, "pages_total": int, "error": str | None}.
    On failure "error" is set and "text" is empty; callers should not ingest.
    """
    ok, detail = health_check()
    if not ok:
        return {"text": "", "pages_ocred": 0, "pages_total": 0, "error": f"OCR backend unavailable: {detail}"}

    images, total_pages = _pdf_to_images(pdf_bytes, max_pages)
    if not images:
        return {"text": "", "pages_ocred": 0, "pages_total": total_pages, "error": "PDF has no pages"}

    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
    results = generate_vllm(batch, max_output_tokens=_max_output_tokens())

    parts = []
    for i, result in enumerate(results, start=1):
        if result.error or not result.raw:
            continue
        markdown = parse_markdown(result.raw, include_headers_footers=True, include_images=False).strip()
        if markdown:
            parts.append(f"--- Page {i} ---\n{markdown}")

    return {
        "text": "\n\n".join(parts),
        "pages_ocred": len(images),
        "pages_total": total_pages,
        "error": None,
    }
