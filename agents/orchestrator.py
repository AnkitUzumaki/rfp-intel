"""Orchestrator / Planner: builds the execution plan, routes work to agents, decides retries and when the job
is done. Extraction planning is deterministic (the field rubric already defines the sub-tasks); question
planning uses the LLM (intent, target bids, queries)."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import prompt
from agents.schemas import FieldDraft, FieldSpec, ValidationResult, groups
from agents.tracing import Span
from settings import cfg


def plan_extraction(specs: dict[str, FieldSpec], manifest: dict[str, dict[str, Any]],
                    addenda: list[int | None]) -> list[str]:
    by_group: dict[str, list[str]] = {}
    for s in specs.values():
        by_group.setdefault(s.group, []).append(s.name)
    docs = ", ".join(f"{m['doc_type']}:{name}" for name, m in manifest.items())
    return [
        f"1. Ingestion Agent: index {len(manifest)} files ({docs})",
        "2. Extraction specialists in parallel, each using the Retrieval Agent's search tool: "
        + "; ".join(f"{groups().get(g, g)} -> {', '.join(f)}" for g, f in by_group.items()),
        f"3. Addendum Reconciliation Agent: addenda {addenda or 'none'}",
        f"4. Validator: evidence/format/consistency/judge; retry failed fields up to "
        f"{cfg('agents', 'max_retries')} times",
        "5. Report Agent: Bid Summary, JSON record, markdown summary",
    ]


def group_tasks(fields: list[str], specs: dict[str, FieldSpec]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in fields:
        out.setdefault(specs[f].group, []).append(f)
    return out


def select_retries(results: dict[str, ValidationResult], drafts: dict[str, FieldDraft], retries: dict[str, int],
                   ) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Feedback loop policy: failed fields retry up to max_retries; not-found fields get one recall retry."""
    max_r = cfg("agents", "max_retries")
    retry, feedback, counts = [], {}, dict(retries)
    for name, r in results.items():
        n = counts.get(name, 0)
        if r.status == "failed" and n < max_r:
            retry.append(name)
            prev = json.dumps(drafts[name].value) if drafts[name].value is not None else "none"
            feedback[name] = f"previous value {prev} rejected: " + "; ".join(r.reasons)
        elif r.status == "not_found" and n == 0 and cfg("agents", "retry_not_found"):
            retry.append(name)
            feedback[name] = "value not found in retrieved passages; search with other terminology"
    for name in retry:
        counts[name] = counts.get(name, 0) + 1
    return retry, feedback, counts


# ---------- question mode ----------

class QAPlan(BaseModel):
    intent: Literal["fact", "compare", "addendum_diff", "extract_field"] = "fact"
    bids: list[str] = Field(default_factory=list)
    addendum_number: int | None = None
    queries: list[str] = Field(default_factory=list)


def bid_catalog(manifests: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = []
    for bid, files in manifests.items():
        info: list[str] = []
        for name, m in files.items():
            fields = m.get("fields") or {}
            for key in ("Solicitation Number", "Title", "Issuing Organization"):
                if fields.get(key):
                    info.append(f"{key}: {fields[key]}")
            if m.get("title"):
                info.append(f"{m['doc_type']} '{m['title']}'")
            info.append(f"file {name} ({m['doc_type']}"
                        + (f" {m['addendum_number']}" if m.get("addendum_number") is not None else "") + ")")
        lines.append(f"- {bid}: " + "; ".join(dict.fromkeys(info)))
    return "\n".join(lines)


def plan_question(question: str, manifests: dict[str, dict[str, dict[str, Any]]], span: Span | None,
                  llm: LLM) -> QAPlan:
    try:
        plan = llm.json(system=prompt("qa_planner_system"),
                        user=prompt("qa_planner_user", catalog=bid_catalog(manifests), question=question),
                        schema=QAPlan, span=span, small=True, purpose="plan_question")
    except (LLMError, GuardrailError):
        plan = QAPlan()  # degrade: search every bid with the raw question
    plan.bids = [b for b in plan.bids if b in manifests] or list(manifests)
    plan.queries = list(dict.fromkeys([question] + [q for q in plan.queries if q.strip()]))[:4]
    return plan
