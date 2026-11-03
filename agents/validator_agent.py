"""Validator / Critic Agent: evidence support, format, cross-field consistency, hallucination checks,
confidence scores. Failed fields go back to retrieval/extraction (the orchestrator enforces max retries)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import prompt, quote_match
from agents.schemas import NOT_FOUND, FieldDraft, FieldSpec, JudgeResponse, ValidationResult
from agents.tracing import Span
from search.retrieve import Evidence
from settings import cfg

_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]{2,5}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NEGATIVE = re.compile(r"^(none|not required|no|n/?a)\b", re.I)
_MONTH = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATES = [(r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"), (r"\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y"),
          (r"[A-Za-z]+ \d{1,2}, \d{4}", "%B %d, %Y"), (r"\d{1,2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y")]
QUOTE_MIN = 0.8
UNSUPPORTED = ("no citation", "quote not found in cited source", "judge: evidence does not support value")


def format_problems(spec: FieldSpec, value: Any) -> list[str]:
    if spec.negative and isinstance(value, str) and _NEGATIVE.match(value.strip()):
        return []
    fmt = spec.format
    if fmt == "datetime" and not (isinstance(value, str) and _DATETIME.match(value.strip())):
        return ["format: expected 'YYYY-MM-DD HH:MM TZ'"]
    if fmt == "date_or_window" and not re.search(rf"\d|{_MONTH}|day|week|month|award", str(value), re.I):
        return ["format: expected a date or a delivery window"]
    if fmt == "enum_list":
        items = value if isinstance(value, list) else [value]
        bad = [v for v in items if str(v) not in spec.enum]
        return [f"format: values must be from {spec.enum}, got {bad}"] if bad else []
    if fmt == "list" and not isinstance(value, list):
        return ["format: expected a list"]
    if fmt == "object":
        if not isinstance(value, dict):
            return [f"format: expected an object with keys {spec.keys}"]
        extra = set(value) - set(spec.keys)
        if extra:
            return [f"format: unexpected keys {sorted(extra)}; allowed {spec.keys}"]
        email = value.get("email")
        if spec.name == "contact_info" and email and not _EMAIL.match(str(email)):
            return [f"format: invalid email {email!r}"]
    if fmt == "text" and not str(value).strip():
        return ["format: empty text"]
    return []


def parse_date(text: str) -> datetime | None:
    for rx, fmt in _DATES:
        m = re.search(rx, str(text))
        if m:
            try:
                return datetime.strptime(m.group(), fmt)
            except ValueError:
                continue
    return None


def consistency_problems(drafts: dict[str, FieldDraft], bid_text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    due, pre = drafts.get("Due Date"), drafts.get("Pre Bid Meeting")
    if due and pre and due.status == pre.status == "found" and isinstance(pre.value, dict):
        d_due, d_pre = parse_date(str(due.value)), parse_date(str(pre.value.get("date", "")))
        if d_due and d_pre and d_pre > d_due:
            out.setdefault("Pre Bid Meeting", []).append("consistency: pre-bid meeting is after the due date")
    norm = re.sub(r"[^a-z0-9]", "", bid_text.lower())
    for name in ("Part_no", "Model_no"):
        d = drafts.get(name)
        if d and d.status == "found" and isinstance(d.value, list):
            missing = [v for v in d.value if re.sub(r"[^a-z0-9]", "", str(v).lower()) not in norm]
            if missing:
                out.setdefault(name, []).append(f"hallucination: {missing} not present in retrieved evidence")
    return out


def _judge(items: dict[str, FieldDraft], span: Span | None, llm: LLM) -> dict[str, tuple[bool, str]] | None:
    if not items or not cfg("agents", "judge_enabled"):
        return None
    lines = [f"- field: {f}\n  value: {json.dumps(d.value)}\n  quotes: " + " | ".join(c.quote for c in d.citations)
             for f, d in items.items()]
    try:
        resp = llm.json(system=prompt("judge_system"), user=prompt("judge_user", items="\n".join(lines)),
                        schema=JudgeResponse, span=span, small=True, purpose="judge")
    except (LLMError, GuardrailError):
        return None  # judge unavailable -> neutral score, rule checks still apply
    return {v.field: (v.supported, v.reason) for v in resp.verdicts if v.field in items}


def validate(drafts: dict[str, FieldDraft], specs: dict[str, FieldSpec], chunks: dict[str, Evidence],
             field_evidence: dict[str, list[Evidence]], span: Span | None, llm: LLM) -> dict[str, ValidationResult]:
    w = cfg("agents", "confidence_weights")
    bid_text = "\n".join(e.text for e in chunks.values())
    consistency = consistency_problems(drafts, bid_text)
    results: dict[str, ValidationResult] = {}
    checks: dict[str, dict[str, float]] = {}
    for name, d in drafts.items():
        spec = specs[name]
        top = max((e.rerank_score or 0.0 for e in field_evidence.get(name, [])), default=0.0)
        if d.status == "error":
            results[name] = ValidationResult(field=name, status="failed", reasons=[d.notes or "extraction error"])
            continue
        if d.status == "not_found":
            # confident "not found" when even the best retrieved passage looked irrelevant
            results[name] = ValidationResult(field=name, status="not_found", reasons=[NOT_FOUND],
                                             confidence=round(0.5 + 0.5 * (1 - top), 2), checks={"top_rerank": top})
            continue
        reasons: list[str] = []
        if not d.citations:
            reasons.append("no citation")
        qm = max((quote_match(c.quote, chunks[c.chunk_id].text) for c in d.citations if c.chunk_id in chunks),
                 default=0.0)
        if d.citations and qm < QUOTE_MIN:
            reasons.append("quote not found in cited source")
        reasons += format_problems(spec, d.value) + consistency.get(name, [])
        cited = [chunks[c.chunk_id].rerank_score or 0.0 for c in d.citations if c.chunk_id in chunks]
        checks[name] = {"quote_match": round(qm, 3), "rerank": round(max(cited, default=top), 3),
                        "agreement": min(1.0, len({c.chunk_id for c in d.citations}) / 2)}
        results[name] = ValidationResult(field=name, status="failed" if reasons else "passed", reasons=reasons)

    verdicts = _judge({n: drafts[n] for n, r in results.items() if r.status == "passed"}, span, llm)
    for name, c in checks.items():
        r = results[name]
        judge = 0.7
        if verdicts is not None and name in verdicts:
            ok, why = verdicts[name]
            judge = 1.0 if ok else 0.0
            if not ok:
                r.reasons.append(f"judge: evidence does not support value ({why})")
                r.status = "failed"
        c["judge"] = judge
        penalty = w["retry_penalty"] * (drafts[name].attempt - 1)
        conf = (w["quote_match"] * c["quote_match"] + w["rerank"] * c["rerank"] + w["agreement"] * c["agreement"]
                + w["judge"] * judge - penalty)
        r.confidence = round(max(0.0, min(1.0, conf if r.status == "passed" else conf * 0.5)), 2)
        r.checks = c
    return results


def is_unsupported(result: ValidationResult) -> bool:
    """Failure means the value may be invented -> never publish it."""
    return any(r.startswith(UNSUPPORTED) or r.startswith("hallucination") for r in result.reasons)
