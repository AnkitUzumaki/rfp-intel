"""OCR fallback for pages without a text layer (scans, image-only tables)."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _engine() -> Any:
    from rapidocr_onnxruntime import RapidOCR  # lazy: model load is slow, only needed for scans

    return RapidOCR()


def ocr_png(png_bytes: bytes) -> str:
    """Return recognized text lines top-to-bottom, or '' when OCR fails."""
    try:
        result, _ = _engine()(png_bytes)
    except Exception as exc:  # OCR must never break ingestion
        log.warning("ocr failed: %s", exc)
        return ""
    if not result:
        return ""
    # result rows: [box, text, score]; sort by top-left y then x
    rows = sorted(result, key=lambda r: (round(r[0][0][1] / 10), r[0][0][0]))
    return "\n".join(r[1] for r in rows)
