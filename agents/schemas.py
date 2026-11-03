"""Structured contracts for every agent-to-agent message (Pydantic) + the field rubric loader."""
from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from settings import ROOT

NOT_FOUND = "Not found in documents"


# ---------- field rubric (config/fields.yaml) ----------

class FieldSpec(BaseModel):
    name: str
    group: str
    description: str
    format: str
    queries: list[str]
    addendum: bool = False
    negative: bool = False
    keys: list[str] = Field(default_factory=list)
    enum: list[str] = Field(default_factory=list)
    scoring: str = "fuzzy"


@lru_cache(maxsize=1)
def _rubric() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / "fields.yaml").read_text())


def load_fields() -> dict[str, FieldSpec]:
    return {name: FieldSpec(name=name, **spec) for name, spec in _rubric()["fields"].items()}


def groups() -> dict[str, str]:
    return dict(_rubric()["groups"])


SUMMARY_FIELD: str = "Bid Summary"


# ---------- LLM response schemas ----------

class Quote(BaseModel):
    evidence_id: str = Field(description="Evidence id such as 'E3' exactly as given in the prompt")
    quote: str = Field(description="Short verbatim span copied from that evidence passage")


class ExtractedField(BaseModel):
    field: str
    found: bool
    value: Any = None
    quotes: list[Quote] = Field(default_factory=list)
    notes: str = ""


class ExtractionResponse(BaseModel):
    fields: list[ExtractedField]


class ChangeItem(BaseModel):
    field: str = Field(description="Exact field name it changes, or 'other' for changes to no listed field")
    addendum_number: int | None = None
    old_value: Any = None
    new_value: Any = None
    description: str
    quote: Quote


class ReconciliationResponse(BaseModel):
    changes: list[ChangeItem] = Field(default_factory=list)


class JudgeItem(BaseModel):
    field: str
    supported: bool
    reason: str = ""


class JudgeResponse(BaseModel):
    verdicts: list[JudgeItem]


class QueryRewrite(BaseModel):
    queries: list[str] = Field(description="2-4 short search queries")


class ClassifyResponse(BaseModel):
    doc_type: Literal["rfp", "addendum", "specs", "affidavit", "bid_page", "other"]
    addendum_number: int | None = None


class AnswerResponse(BaseModel):
    answer: str = Field(description="Answer with [E#] citations, or 'Not found in documents.'")


class GoNoGoExplanation(BaseModel):
    explanation: str


class SummaryResponse(BaseModel):
    summary: str = Field(description="3-6 sentence summary of the bid")
    fields_used: list[str] = Field(default_factory=list)


# ---------- internal messages / state ----------

class Citation(BaseModel):
    chunk_id: str
    file: str
    page: int
    quote: str = ""

    def source(self) -> dict[str, Any]:
        return {"file": self.file, "page": self.page}


class FieldDraft(BaseModel):
    field: str
    value: Any = None
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""
    status: Literal["found", "not_found", "error"] = "not_found"
    attempt: int = 1


class ValidationResult(BaseModel):
    field: str
    status: Literal["passed", "failed", "not_found"]
    reasons: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    checks: dict[str, float] = Field(default_factory=dict)


class AddendumChange(BaseModel):
    field: str
    addendum_number: int | None
    old_value: Any = None
    new_value: Any = None
    description: str
    source: Citation


# ---------- final output (PDF §8.1) ----------

class FieldOutput(BaseModel):
    value: Any = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


class BidRecord(BaseModel):
    bid_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    run_id: str = ""
    fields: dict[str, FieldOutput]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: dict[str, int] = Field(default_factory=dict)
    processing_errors: list[str] = Field(default_factory=list)


class QAAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    answered_by: str = ""
    bids: list[str] = Field(default_factory=list)
    run_id: str = ""
