"""Q&A / Report Agent (report side): Bid Summary, final JSON record (PDF §8.1) and a human-readable summary."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import prompt
from agents.schemas import (
    NOT_FOUND,
    SUMMARY_FIELD,
    AddendumChange,
    BidRecord,
    FieldDraft,
    FieldOutput,
    FieldSpec,
    SummaryResponse,
    ValidationResult,
)
from agents.tracing import Span
from agents.validator_agent import is_unsupported
from settings import ROOT, cfg


def _sources(draft: FieldDraft) -> list[dict[str, Any]]:
    return list({(c.file, c.page): c.source() for c in draft.citations}.values())


def field_output(draft: FieldDraft, v: ValidationResult) -> FieldOutput:
    if v.status == "not_found":
        return FieldOutput(value=None, confidence=v.confidence, notes=draft.notes or NOT_FOUND)
    if v.status == "failed" and (is_unsupported(v) or draft.value is None):
        # guardrail: an unverified value is never published
        return FieldOutput(value=None, confidence=v.confidence,
                           notes=f"{NOT_FOUND} (unverified candidate rejected: {'; '.join(v.reasons)})")
    notes = draft.notes
    if v.status == "failed":
        notes = (notes + " | " if notes else "") + "Validation warning: " + "; ".join(v.reasons)
    return FieldOutput(value=draft.value, sources=_sources(draft), confidence=v.confidence, notes=notes)


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def summarize(bid_id: str, outputs: dict[str, FieldOutput], changes: list[AddendumChange], span: Span | None,
              llm: LLM) -> FieldOutput:
    known = {k: v.value for k, v in outputs.items() if v.value is not None}
    try:
        resp = llm.json(system=prompt("summary_system"),
                        user=prompt("summary_user", bid_id=bid_id, fields=json.dumps(known, indent=1),
                                    changes="\n".join(f"- Addendum {c.addendum_number}: {c.description}"
                                                      for c in changes) or "none"),
                        schema=SummaryResponse, span=span, purpose="summary")
        text, used = resp.summary.strip(), [f for f in resp.fields_used if f in outputs]
    except (LLMError, GuardrailError):
        # deterministic fallback from validated values
        text = " ".join(f"{k}: {json.dumps(v)}." for k, v in list(known.items())[:5])
        used = list(known)[:5]
    sources = list({(s["file"], s["page"]): s for f in (used or list(known)) for s in outputs[f].sources}.values())
    n = sentence_count(text)
    ok = 3 <= n <= 6 and bool(sources)
    confidence = round(sum(outputs[f].confidence for f in used) / len(used), 2) if used else 0.5
    return FieldOutput(value=text or None, sources=sources, confidence=confidence if ok else confidence * 0.5,
                       notes="Generated from validated fields" + ("" if ok else f" (warning: {n} sentences)"))


def build_record(bid_id: str, run_id: str, specs: dict[str, FieldSpec], drafts: dict[str, FieldDraft],
                 validation: dict[str, ValidationResult], changes: list[AddendumChange], errors: list[str],
                 span: Span | None, llm: LLM) -> BidRecord:
    outputs = {name: field_output(drafts[name], validation[name]) for name in specs}
    outputs[SUMMARY_FIELD] = summarize(bid_id, outputs, changes, span, llm)
    counts = {"passed": 0, "failed": 0, "not_found": 0}
    for name in specs:
        counts[validation[name].status] += 1
    counts["passed" if outputs[SUMMARY_FIELD].value else "failed"] += 1
    return BidRecord(bid_id=bid_id, run_id=run_id, fields=outputs, addendum_changes=changes, validation=counts,
                     processing_errors=errors)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_markdown(record: BidRecord) -> str:
    v = record.validation
    lines = [f"# {record.bid_id} — Bid Report", "",
             f"Run `{record.run_id}` · generated {record.generated_at} · "
             f"validation: {v.get('passed', 0)} passed, {v.get('failed', 0)} failed, {v.get('not_found', 0)} not found",
             "", "## Summary", "", _fmt(record.fields.get(SUMMARY_FIELD, FieldOutput()).value), "",
             "## Fields", "", "| Field | Value | Confidence | Sources | Notes |", "|---|---|---|---|---|"]
    for name, f in record.fields.items():
        if name == SUMMARY_FIELD:
            continue
        src = "; ".join(f"{s['file']} p.{s['page']}" for s in f.sources) or "—"
        lines.append(f"| {name} | {_fmt(f.value).replace('|', '/')} | {f.confidence:.2f} | {src} | "
                     f"{f.notes.replace('|', '/')} |")
    lines += ["", "## Addendum changes", ""]
    lines += [f"- **Addendum {c.addendum_number}** ({c.field}): {c.description} — _{c.source.file} p.{c.source.page}_"
              for c in record.addendum_changes] or ["None found."]
    lines += ["", "## Processing notes", ""] + ([f"- {e}" for e in record.processing_errors] or ["None."])
    return "\n".join(lines) + "\n"


def save_outputs(record: BidRecord, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or ROOT / cfg("outputs", "dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{record.bid_id}.json", out_dir / f"{record.bid_id}.md"
    json_path.write_text(record.model_dump_json(indent=2))
    md_path.write_text(render_markdown(record))
    return json_path, md_path
