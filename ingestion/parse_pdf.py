"""PDF parsing: per-page text + tables (PyMuPDF), OCR fallback, failure handling."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf

from ingestion.models import Page
from ingestion.ocr import ocr_png

log = logging.getLogger(__name__)


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cells = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    cells = [r for r in cells if any(r)]
    if not cells:
        return ""
    width = max(len(r) for r in cells)
    cells = [r + [""] * (width - len(r)) for r in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines)


def _table_md(table: pymupdf.table.Table) -> str:
    # to_markdown keeps cell text intact; extract() can drop spaces/reorder chars on some PDFs
    try:
        md = table.to_markdown(clean=False).replace("<br>", " ").strip()
        head, _, rest = md.partition("\n")
        md = re.sub(r"Col\d+", "", head) + "\n" + rest  # PyMuPDF placeholder names for header-less tables
        return md if md.count("|") > 2 else ""
    except Exception:
        return table_to_markdown(table.extract())


def _inside(block: tuple, boxes: list[pymupdf.Rect]) -> bool:
    r = pymupdf.Rect(block[:4])
    return any(b.contains(r) or (b & r).get_area() > 0.6 * r.get_area() for b in boxes)


def _parse_page(page: pymupdf.Page, ocr_enabled: bool, min_chars: int, ocr_dpi: int) -> tuple[Page, str | None]:
    """Return the parsed page and an optional problem note for the error log."""
    try:
        found = page.find_tables().tables
    except Exception as exc:  # table detection is best-effort
        log.warning("table detection failed p%s: %s", page.number + 1, exc)
        found = []
    boxes = [pymupdf.Rect(t.bbox) for t in found]
    tables = [md for md in (_table_md(t) for t in found) if md]
    # text outside table regions, so tables aren't indexed twice
    blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and not _inside(b, boxes)]
    text = "\n".join(b[4] for b in blocks)

    widgets = [f"{w.field_name}: {w.field_value}" for w in page.widgets() if w.field_name and w.field_value]
    if widgets:  # fillable form values are not part of the text layer
        text = text + "\n" + "\n".join(widgets)
    if len(text.strip()) + sum(map(len, tables)) >= min_chars:
        return Page(page_number=page.number + 1, text=text, tables=tables), None
    # no text layer: scanned images, or text drawn as vector outlines -> render + OCR
    visual = bool(page.get_images()) or len(page.get_drawings()) > 20
    if visual and ocr_enabled:
        ocr_text = ocr_png(page.get_pixmap(dpi=ocr_dpi).tobytes("png"))
        if ocr_text.strip():
            return Page(page_number=page.number + 1, text=ocr_text, ocr=True), f"page {page.number + 1}: ocr"
        return Page(page_number=page.number + 1), f"page {page.number + 1}: scanned, ocr found no text"
    if visual:
        return Page(page_number=page.number + 1), f"page {page.number + 1}: scanned, ocr disabled"
    return Page(page_number=page.number + 1), f"page {page.number + 1}: empty"


def parse_pdf(
    path: Path, ocr_enabled: bool = True, min_chars: int = 20, ocr_dpi: int = 150
) -> tuple[list[Page], dict, list[str]]:
    """Parse a PDF into pages. Never raises: problems are returned as error strings."""
    errors: list[str] = []
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        return [], {}, [f"unreadable pdf: {exc}"]
    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            return [], {}, ["encrypted pdf: password required"]
        pages: list[Page] = []
        for i in range(doc.page_count):
            page = doc[i]
            try:
                parsed, note = _parse_page(page, ocr_enabled, min_chars, ocr_dpi)
            except Exception as exc:
                parsed, note = Page(page_number=page.number + 1), f"page {page.number + 1}: parse error {exc}"
            pages.append(parsed)
            if note:
                errors.append(note)
        return pages, dict(doc.metadata or {}), errors
