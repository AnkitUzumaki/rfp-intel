"""Ingestion pipeline: bid folder -> classified, cleaned Documents -> Chunks."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from ingestion.chunk import chunk_document
from ingestion.classify import classify, pdf_date
from ingestion.clean import clean_text, remove_headers_footers
from ingestion.detect import detect_file_type
from ingestion.models import Chunk, Document, Page
from ingestion.parse_html import parse_html
from ingestion.parse_pdf import parse_pdf
from settings import cfg

log = logging.getLogger(__name__)

# Bump when parsing/cleaning/chunking output changes: indexed files are re-ingested on mismatch.
PIPELINE_VERSION = "2"


def _pdf_title(meta: dict, pages: list[Page]) -> str | None:
    title = re.sub(r"^Microsoft (Word|Excel) - ", "", (meta.get("title") or "").strip())
    if title or not pages:
        return title or None
    return next((ln.strip() for ln in pages[0].text.splitlines() if len(ln.strip()) > 5), None)


def ingest_file(path: Path, bid_id: str) -> Document:
    data = path.read_bytes()
    file_type = detect_file_type(path)
    doc = Document(
        bid_id=bid_id, file_name=path.name, file_path=str(path), file_type=file_type,
        content_hash=hashlib.sha256(data).hexdigest(),
    )
    first_raw = ""
    if file_type == "pdf":
        pages, meta, errors = parse_pdf(
            path, cfg("ingestion", "ocr_enabled"), cfg("ingestion", "min_page_chars"), cfg("ingestion", "ocr_dpi")
        )
        first_raw = pages[0].text if pages else ""
        texts = remove_headers_footers(
            [p.text for p in pages], cfg("ingestion", "header_footer_ratio"), cfg("ingestion", "header_footer_zone")
        )
        doc.pages = [p.model_copy(update={"text": clean_text(t)}) for p, t in zip(pages, texts, strict=True)]
        doc.doc_date, doc.title, doc.errors = pdf_date(meta), _pdf_title(meta, doc.pages), errors
    elif file_type == "html":
        text, title, fields, errors = parse_html(path)
        first_raw = text
        doc.pages = [Page(page_number=1, text=text)]
        doc.title, doc.fields, doc.errors = title, fields, errors
    else:
        doc.errors = ["unsupported file type"]
    doc.doc_type, doc.addendum_number = classify(path.name, file_type, first_raw)
    for err in doc.errors:
        log.warning("ingest %s/%s: %s", bid_id, path.name, err)
    return doc


def ingest_folder(folder: Path) -> list[Document]:
    """One folder = one bid; bid_id = folder name. Hidden files are skipped."""
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    return [ingest_file(p, folder.name) for p in files]


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [
        c for d in docs
        for c in chunk_document(d, cfg("chunking", "target_chars"), cfg("chunking", "overlap_chars"),
                                cfg("chunking", "max_table_chars"))
    ]
