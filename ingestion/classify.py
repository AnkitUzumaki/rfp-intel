"""Document classification (doc_type, addendum number, date) from file name + first-page text.

Rules only; no bid-specific values. Unmatched files become 'other' (the ingestion agent may
ask the LLM to reclassify those).
"""
from __future__ import annotations

import re

from ingestion.models import DocType

_ADDENDUM_NO = re.compile(r"\baddend(?:um|a)\b[\s_\-]*(?:no\.?|number|#)?[\s_\-]*(\d+)", re.I)
_OTHER_NAME = re.compile(r"question|\bq\s*&\s*a\b", re.I)
_NAME_RULES: list[tuple[DocType, re.Pattern[str]]] = [
    ("addendum", re.compile(r"addend(um|a)", re.I)),
    ("affidavit", re.compile(r"affidavit", re.I)),
    ("specs", re.compile(r"spec(s|ification)?|datasheet|quote", re.I)),
    ("rfp", re.compile(r"\b(rfp|rfq|ifb|itb|porfp|solicitation)\b|request.for", re.I)),
]
_TEXT_RULES: list[tuple[DocType, re.Pattern[str], int]] = [  # (type, pattern, chars of page 1 inspected)
    ("addendum", re.compile(r"\baddend(?:um|a)\b\s*(?:no\.?|number|#)\s*\d+", re.I), 400),
    ("affidavit", re.compile(r"\baffidavit\b", re.I), 400),
    ("rfp", re.compile(r"request\s+for\s+(proposals?|quotes?|quotations?)|invitation\s+(for|to)\s+bid"
                       r"|\b(rfp|rfq|ifb|porfp)\b", re.I), 1500),
    ("specs", re.compile(r"\bsku\b|specifications?|technical\s+requirements", re.I), 1500),
]


def classify(file_name: str, file_type: str, first_text: str) -> tuple[DocType, int | None]:
    if file_type == "html":
        return "bid_page", None
    name = re.sub(r"[_\-]+", " ", file_name.rsplit(".", 1)[0])
    doc_type: DocType = "other"
    if not _OTHER_NAME.search(name):
        doc_type = next((t for t, rx in _NAME_RULES if rx.search(name)), "other")
        if doc_type == "other":
            doc_type = next((t for t, rx, n in _TEXT_RULES if rx.search(first_text[:n])), "other")
    number = None
    if doc_type == "addendum":
        m = _ADDENDUM_NO.search(name) or _ADDENDUM_NO.search(first_text[:400])
        number = int(m.group(1)) if m else None
    return doc_type, number


def pdf_date(meta: dict) -> str | None:
    """PDF metadata date 'D:20240613175856-05'00'' -> '2024-06-13'."""
    for key in ("creationDate", "modDate"):
        m = re.match(r"D:(\d{4})(\d{2})(\d{2})", meta.get(key) or "")
        if m:
            return "-".join(m.groups())
    return None
