"""RFP-aware chunking: heading-aware text packing with overlap; tables as atomic chunks.

Chunks never cross pages, so every chunk cites exactly one page.
"""
from __future__ import annotations

import re
import uuid

from ingestion.models import Chunk, Document

_HEADING = re.compile(
    r"^(## .+"
    r"|(Section|SECTION|Article|ARTICLE|Part|PART|Attachment|ATTACHMENT|Exhibit|EXHIBIT|Appendix|APPENDIX)\b.{0,80}"
    r"|\d+(\.\d+)*\.?\s+[A-Z][^\n]{2,70}"
    r"|[A-Z][A-Z0-9 &/,\-()']{3,70})$"
)


def is_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 90 and bool(_HEADING.match(line))


def _units(text: str) -> list[tuple[str | None, str]]:
    """Split page text into (heading, body) units at heading lines."""
    units: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        if is_heading(line):
            units.append((line.lstrip("# ").strip(), [line]))
        else:
            units[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in units if "\n".join(b).strip()]


def _windows(text: str, target: int, overlap: int) -> list[str]:
    """Split long text on line boundaries into ~target-size windows overlapping by ~overlap chars."""
    lines = text.splitlines()
    out, start = [], 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and (size == 0 or size + len(lines[end]) <= target):
            size += len(lines[end]) + 1
            end += 1
        out.append("\n".join(lines[start:end]))
        if end >= len(lines):
            break
        back, kept = end, 0
        while back > start + 1 and kept + len(lines[back - 1]) <= overlap:
            back -= 1
            kept += len(lines[back]) + 1
        start = back
    return out


def split_table(md: str, max_chars: int) -> list[str]:
    """Split a markdown table by rows, repeating the header in every part."""
    if len(md) <= max_chars:
        return [md]
    lines = md.splitlines()
    head, rows = lines[:2], lines[2:]
    parts, cur = [], list(head)
    for row in rows:
        if len("\n".join(cur + [row])) > max_chars and len(cur) > 2:
            parts.append("\n".join(cur))
            cur = list(head)
        cur.append(row)
    parts.append("\n".join(cur))
    return parts


def chunk_document(doc: Document, target: int = 2000, overlap: int = 300, max_table: int = 3000) -> list[Chunk]:
    chunks: list[Chunk] = []
    section: str | None = None

    def add(page_no: int, text: str, sec: str | None, is_table: bool) -> None:
        cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.content_hash}:{page_no}:{len(chunks)}"))
        chunks.append(Chunk(
            chunk_id=cid, bid_id=doc.bid_id, file_name=doc.file_name, doc_type=doc.doc_type,
            addendum_number=doc.addendum_number, doc_date=doc.doc_date, page_number=page_no,
            section=sec, is_table=is_table, text=text,
        ))

    for page in doc.pages:
        buf, buf_sec = "", section
        for heading, body in _units(page.text):
            if heading:
                section = heading
            if buf and len(buf) + len(body) > target:
                add(page.page_number, buf, buf_sec, False)
                buf, buf_sec = "", section
            if len(body) > target:
                for w in _windows(body, target, overlap):
                    add(page.page_number, w, section, False)
                continue
            buf = f"{buf}\n\n{body}" if buf else body
            buf_sec = buf_sec or section
        if buf:
            add(page.page_number, buf, buf_sec, False)
        for table in page.tables:
            for part in split_table(table, max_table):
                add(page.page_number, part, section, True)
    return chunks
