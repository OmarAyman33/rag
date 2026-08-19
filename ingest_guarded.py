"""ingest_guarded.py - guarded ingestion CLI (NEW code, original repo untouched).

Adds to the original ingestion:
  * PDF support (pypdf) - the original only accepted .txt and .md
  * PII gate (P): chunks containing personal information are BLOCKED (default),
    REDACTED, or REPORTED depending on --pii-mode (Sovereign-approved policy:
    block fundamentally personal chunks, redact incidental identifiers)
  * Configurable paths (original hardcoded /home/omar/spectech/RAG/...)

Usage:
  python ingest_guarded.py --input-dir input
  python ingest_guarded.py --input-dir input --pii-mode redact --verbose
  python ingest_guarded.py --file input/some.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from guardrails.config import load_settings
from guardrails.pii import classify_chunk

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def read_document(path: Path) -> str:
    """Read a .txt/.md/.pdf file into plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        if not text.strip():
            raise ValueError(f"PDF '{path.name}' produced no extractable text")
        return text
    return path.read_text(encoding="utf-8", errors="replace")


def ingest(
    input_dir: Path,
    settings,
    splitter: RecursiveCharacterTextSplitter,
    files: list[Path] | None = None,
) -> dict:
    """Ingest documents into Chroma through the PII gate.

    Returns a report dict: files processed, chunks added, chunks blocked/redacted.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = chroma_client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embed_model = SentenceTransformer(settings.embed_model)

    if files is None:
        files = sorted(
            p for p in Path(input_dir).iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )

    report = {
        "files": [],
        "chunks_added": 0,
        "chunks_blocked": 0,
        "chunks_redacted": 0,
        "chunks_clean": 0,
        "errors": [],
    }

    def already_ingested(document_path: Path) -> bool:
        existing = collection.get(where={"source": document_path.name}, limit=1)
        return len(existing["ids"]) > 0

    for document_path in files:
        entry = {"name": document_path.name, "status": "ok", "chunks": 0,
                 "blocked": 0, "redacted": 0, "reason": ""}
        try:
            if already_ingested(document_path):
                entry["status"] = "skipped"
                entry["reason"] = "already ingested"
                report["files"].append(entry)
                continue

            document = read_document(document_path)
            chunks = splitter.split_text(document)
            if not chunks:
                entry["status"] = "empty"
                entry["reason"] = "no chunks produced"
                report["files"].append(entry)
                continue

            clean_chunks: list[str] = []
            clean_ids: list[str] = []
            clean_metas: list[dict] = []

            for num, chunk in enumerate(chunks, start=1):
                classification = classify_chunk(chunk, mode=settings.pii_mode)

                if classification.action == "block":
                    report["chunks_blocked"] += 1
                    entry["blocked"] += 1
                    entry["reason"] = classification.reason
                    if settings.verbose:
                        print(f"  [PII BLOCK] {document_path.name} chunk {num}: "
                              f"{classification.reason}")
                    continue

                text = (
                    classification.redacted_text
                    if classification.action == "redact"
                    else chunk
                )
                if classification.action == "redact":
                    report["chunks_redacted"] += 1
                    entry["redacted"] += 1
                    if settings.verbose:
                        print(f"  [PII REDACT] {document_path.name} chunk {num}")
                else:
                    report["chunks_clean"] += 1

                clean_chunks.append(text)
                clean_ids.append(f"{document_path.stem}_chunk_{num}")
                clean_metas.append(
                    {"source": document_path.name, "chunk_index": num}
                )

            if clean_chunks:
                embeddings = embed_model.encode_document(clean_chunks).tolist()
                collection.add(
                    ids=clean_ids,
                    documents=clean_chunks,
                    embeddings=embeddings,
                    metadatas=clean_metas,
                )
                report["chunks_added"] += len(clean_chunks)
                entry["chunks"] = len(clean_chunks)

            report["files"].append(entry)
        except Exception as exc:  # keep going; report the error
            entry["status"] = "error"
            entry["reason"] = str(exc)
            report["errors"].append(f"{document_path.name}: {exc}")
            report["files"].append(entry)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs/txt/md into Chroma through the PII gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help="folder to scan for .txt/.md/.pdf files")
    parser.add_argument("--file", type=str, default=None,
                        help="ingest a single file instead of a folder")
    parser.add_argument("--pii-mode", type=str, default=None,
                        choices=["block", "redact", "report"],
                        help="how to treat personal information in chunks")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--force", action="store_true",
                        help="re-ingest even if already ingested")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the report as JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    overrides = {"verbose": args.verbose}
    if args.pii_mode:
        overrides["pii_mode"] = args.pii_mode
    if args.input_dir:
        overrides["input_dir"] = Path(args.input_dir)

    settings = load_settings(overrides)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    files = None
    if args.file:
        files = [Path(args.file)]

    report = ingest(settings.input_dir, settings, splitter, files=files)

    if args.as_json:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"\nIngestion report ({settings.pii_mode} mode):")
    for entry in report["files"]:
        print(f"  {entry['name']}: {entry['status']} "
              f"(chunks={entry['chunks']}, blocked={entry['blocked']}, "
              f"redacted={entry['redacted']}) {entry['reason']}")
    print(f"  TOTAL added={report['chunks_added']} blocked={report['chunks_blocked']} "
          f"redacted={report['chunks_redacted']} clean={report['chunks_clean']}")
    if report["errors"]:
        print("\nErrors:")
        for e in report["errors"]:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())