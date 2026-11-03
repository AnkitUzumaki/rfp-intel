"""Ingestion Agent: detects file types, parses + cleans + chunks (ingestion/), triggers incremental indexing,
and asks the LLM to classify documents the rules could not ('other')."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import prompt
from agents.retrieval_agent import search_tool
from agents.schemas import ClassifyResponse
from agents.tracing import Span
from search.index import SearchIndex, get_index
from search.retrieve import SearchFilters


def _reclassify_other(index: SearchIndex, bid_id: str, manifest: dict[str, dict[str, Any]], span: Span | None,
                      llm: LLM) -> None:
    for name, meta in manifest.items():
        if meta["doc_type"] != "other" or meta.get("reclassified") or not meta.get("pages"):
            continue
        first = search_tool("title and purpose of this document", SearchFilters(bid_id=bid_id, file_name=name),
                            span, top_k=1)
        if not first:
            continue
        try:
            out = llm.json(system=prompt("classify_system"),
                           user=prompt("classify_user", file_name=name, text=first[0].text[:1500]),
                           schema=ClassifyResponse, span=span, small=True, purpose=f"classify:{name}")
        except (LLMError, GuardrailError):
            continue  # stays 'other' — still searchable
        if out.doc_type != "other":
            index.set_doc_meta(bid_id, name, out.doc_type, out.addendum_number)


def run_ingestion(folder: Path, span: Span | None, llm: LLM, index: SearchIndex | None = None,
                  ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index = index or get_index()
    start = time.perf_counter()
    stats = index.index_folder(folder)
    if span is not None:
        span.tool("index_folder", str(folder), {k: v for k, v in stats.items() if k != "bid_id"},
                  (time.perf_counter() - start) * 1000)
    _reclassify_other(index, folder.name, index.manifest(folder.name), span, llm)
    manifest = index.manifest(folder.name)
    # OCR notes are recoveries, not failures; everything else is surfaced in the output JSON
    errors = [f"{name}: {err}" for name, meta in manifest.items() for err in meta.get("errors", [])
              if not err.endswith(": ocr")]
    return manifest, errors
