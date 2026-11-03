"""Addendum Reconciliation Agent: compares base-document values with addenda (retrieved via the search tool),
produces a change log and applies the latest valid change to each affected field."""
from __future__ import annotations

import json
from typing import Any

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import EvidenceBook, prompt
from agents.retrieval_agent import search_tool
from agents.schemas import AddendumChange, FieldDraft, FieldSpec, ReconciliationResponse
from agents.tracing import Span
from search.retrieve import Evidence, SearchFilters

CHANGE_QUERY = "purpose of this addendum changes extended revised amended replaced new date answer"


def addenda_numbers(manifest: dict[str, dict[str, Any]]) -> list[int | None]:
    nums = {m.get("addendum_number") for m in manifest.values() if m.get("doc_type") == "addendum"}
    return sorted(nums, key=lambda n: (n is None, n or 0))


def addendum_evidence(bid_id: str, numbers: list[int | None], specs: list[FieldSpec], span: Span | None,
                      ) -> list[Evidence]:
    pools: list[Evidence] = []
    for n in numbers:
        flt = SearchFilters(bid_id=bid_id, doc_types=["addendum"], addendum_number=n)
        pools += search_tool(CHANGE_QUERY, flt, span, top_k=6)
    for s in specs:
        pools += search_tool(s.queries[0], SearchFilters(bid_id=bid_id, doc_types=["addendum"]), span, top_k=2)
    return list({e.chunk_id: e for e in pools}.values())


def _current(drafts: dict[str, FieldDraft], specs: list[FieldSpec]) -> str:
    lines = []
    for s in specs:
        d = drafts.get(s.name)
        val = json.dumps(d.value) if d and d.status == "found" else "not found yet"
        src = f" (source: {d.citations[0].file} p{d.citations[0].page})" if d and d.citations else ""
        lines.append(f"- {s.name} = {val}{src}  [required shape: {s.description}]")
    return "\n".join(lines)


def apply_changes(drafts: dict[str, FieldDraft], changes: list[AddendumChange]) -> dict[str, FieldDraft]:
    """Latest addendum wins: apply in ascending addendum order. Returns a new dict (input untouched)."""
    out = dict(drafts)
    for ch in sorted(changes, key=lambda c: c.addendum_number or 0):
        d = out.get(ch.field)
        if d is None or ch.new_value in (None, "", [], {}):
            continue
        original = ch.old_value if ch.old_value not in (None, "") else (d.value if d.status == "found" else None)
        cites = [ch.source] + [c for c in d.citations if c.chunk_id != ch.source.chunk_id]
        note = f"Changed by Addendum {ch.addendum_number}" + (f" (original: {original})" if original else "")
        out[ch.field] = d.model_copy(update={"value": ch.new_value, "citations": cites, "status": "found",
                                             "notes": note})
    return out


def reconcile(bid_id: str, drafts: dict[str, FieldDraft], specs: list[FieldSpec], numbers: list[int | None],
              span: Span | None, llm: LLM, include_other: bool = True,
              ) -> tuple[dict[str, FieldDraft], list[AddendumChange], list[Evidence]]:
    specs = [s for s in specs if s.addendum]
    if not numbers or not specs:
        return drafts, [], []
    evidence = addendum_evidence(bid_id, numbers, specs, span)
    if not evidence:
        return drafts, [], []
    book = EvidenceBook()
    book.add_all(evidence)
    try:
        resp = llm.json(system=prompt("reconciliation_system"),
                        user=prompt("reconciliation_user", bid_id=bid_id, current=_current(drafts, specs),
                                    evidence=book.render()),
                        schema=ReconciliationResponse, span=span, purpose="reconcile")
    except (LLMError, GuardrailError):
        return drafts, [], evidence  # base values stand; validator still checks them
    names = {s.name.lower(): s.name for s in specs}
    changes: list[AddendumChange] = []
    for item in resp.changes:
        cite = book.citation(item.quote)
        if cite is None:  # uncited change -> ignored (guardrail: every value needs a source)
            continue
        field = names.get(item.field.strip().lower(), "other")
        if field == "other" and not include_other:
            continue
        number = item.addendum_number
        if number is None:
            number = book.by_id[item.quote.evidence_id.strip("[] ")].addendum_number
        changes.append(AddendumChange(field=field, addendum_number=number, old_value=item.old_value,
                                      new_value=item.new_value, description=item.description, source=cite))
    return apply_changes(drafts, changes), changes, evidence
