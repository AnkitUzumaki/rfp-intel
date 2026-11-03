"""Extraction specialists (one per field group): fill fields from retrieved evidence only, with citations.

Fields with no retrieved evidence are marked not found without an LLM call (empty-retrieval path).
"""
from __future__ import annotations

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import EvidenceBook, prompt
from agents.schemas import NOT_FOUND, ExtractionResponse, FieldDraft, FieldSpec, groups
from agents.tracing import Span
from search.retrieve import Evidence


def _empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def extract_group(group: str, specs: list[FieldSpec], evidence: dict[str, list[Evidence]], bid_id: str,
                  feedback: dict[str, str], attempt: int, span: Span | None, llm: LLM) -> dict[str, FieldDraft]:
    drafts = {s.name: FieldDraft(field=s.name, status="not_found", notes=NOT_FOUND, attempt=attempt) for s in specs}
    ask = [s for s in specs if evidence.get(s.name)]
    if not ask:
        return drafts

    book = EvidenceBook()
    lines = []
    for s in ask:
        ids = book.add_all(evidence[s.name])
        lines.append(f"- {s.name}: {s.description} [most relevant evidence: {', '.join(ids)}]")
    fb = [f"- {f}: {why}" for f, why in feedback.items() if f in {s.name for s in ask}]
    feedback_txt = ("\nA previous attempt was rejected by the validator. Fix these problems:\n" + "\n".join(fb)
                    + "\n") if fb else ""
    try:
        resp = llm.json(
            system=prompt("extraction_system", group_name=groups().get(group, group)),
            user=prompt("extraction_user", bid_id=bid_id, field_specs="\n".join(lines), feedback=feedback_txt,
                        evidence=book.render()),
            schema=ExtractionResponse, span=span, purpose=f"extract:{group}",
        )
    except (LLMError, GuardrailError) as exc:
        for s in ask:
            drafts[s.name] = FieldDraft(field=s.name, status="error", notes=f"extraction failed: {exc}",
                                        attempt=attempt)
        return drafts

    by_name = {s.name.lower(): s.name for s in ask}
    for item in resp.fields:
        name = by_name.get(item.field.strip().lower())
        if name is None:
            continue
        citations = [c for c in (book.citation(q) for q in item.quotes) if c is not None]
        if item.found and not _empty(item.value):
            drafts[name] = FieldDraft(field=name, value=item.value, citations=citations, notes=item.notes,
                                      status="found", attempt=attempt)
        else:
            drafts[name] = FieldDraft(field=name, status="not_found", notes=item.notes or NOT_FOUND, attempt=attempt)
    return drafts
