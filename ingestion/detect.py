"""File type detection by content (magic bytes), not by extension."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

FileType = Literal["pdf", "html", "unknown"]


def detect_file_type(path: Path) -> FileType:
    try:
        head = path.read_bytes()[:2048]
    except OSError:
        return "unknown"
    if head.lstrip().startswith(b"%PDF"):
        return "pdf"
    sniff = head.lower()
    if b"<html" in sniff or b"<!doctype html" in sniff:
        return "html"
    return "unknown"
