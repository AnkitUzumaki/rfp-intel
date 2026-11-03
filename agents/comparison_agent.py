"""Bid Comparison Agent (bonus): side-by-side comparison of several bids on any aspect, with citations.

Used by Q&A mode ("compare ...") and by the full comparison report over all extracted bid records.
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.llm import LLM
from agents.schemas import SUMMARY_FIELD, BidRecord, Citation
from agents.tracing import Span
from settings import ROOT, cfg


def compare_bids(question: str, bids: list[str], queries: list[str], span: Span | None, llm: LLM,
                 ) -> tuple[str, list[Citation]]:
    from agents.qa_agent import answer_from_evidence  # local: qa_agent imports this module
    from agents.retrieval_agent import question_evidence

    per_bid = max(3, cfg("agents", "qa_top_k") // max(1, len(bids)) + 1)
    evidence = question_evidence(queries, bids, span, per_bid)  # balanced: top passages from every bid
    return answer_from_evidence(question, evidence, span, llm, system="comparison_system",
                                context=f"Bids to compare: {', '.join(bids)}")


def _cell(record: BidRecord, field: str) -> str:
    f = record.fields.get(field)
    if f is None or f.value is None:
        return "Not found"
    value = json.dumps(f.value, ensure_ascii=False) if isinstance(f.value, (dict, list)) else str(f.value)
    src = f.sources[0] if f.sources else None
    cite = f" ({src['file']} p.{src['page']})" if src else ""
    return (value[:160] + ("…" if len(value) > 160 else "") + cite).replace("|", "/")


def comparison_report(records: list[BidRecord], out_path: Path | None = None) -> str:
    """Deterministic side-by-side table of all extracted fields (no LLM: values already validated + cited)."""
    fields = [f for f in records[0].fields if f != SUMMARY_FIELD]
    lines = ["# Bid Comparison", "", "| Field | " + " | ".join(r.bid_id for r in records) + " |",
             "|---|" + "---|" * len(records)]
    lines += [f"| {f} | " + " | ".join(_cell(r, f) for r in records) + " |" for f in fields]
    lines += ["", "## Summaries", ""] + [f"**{r.bid_id}:** {r.fields[SUMMARY_FIELD].value}" for r in records]
    text = "\n".join(lines) + "\n"
    out_path = out_path or ROOT / cfg("outputs", "dir") / "comparison.md"
    out_path.write_text(text)
    return text
