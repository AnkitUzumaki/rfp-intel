"""Shared state / memory for the agent graphs (PDF §7.2): plan, evidence, draft fields, validation, output.

Reducers let parallel extraction branches write their own fields into the same state safely.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from agents.schemas import AddendumChange, BidRecord, Citation, FieldDraft, ValidationResult
from search.retrieve import Evidence


def merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class ExtractState(TypedDict, total=False):
    run_id: str
    bid_id: str
    folder: str
    manifest: dict[str, dict[str, Any]]
    addenda: list[int | None]
    plan: list[str]
    pending: list[str]                       # fields to (re)extract in the current round
    feedback: dict[str, str]                 # validator reasons, fed back into retrieval/extraction
    attempt: int
    retries: dict[str, int]
    evidence: Annotated[dict[str, list[Evidence]], merge]   # field -> retrieved passages
    chunks: Annotated[dict[str, Evidence], merge]           # chunk_id -> passage (for quote checks)
    drafts: Annotated[dict[str, FieldDraft], merge]
    changes: Annotated[list[AddendumChange], operator.add]
    validation: dict[str, ValidationResult]
    record: BidRecord
    errors: Annotated[list[str], operator.add]


class GroupTask(TypedDict):
    """Payload sent to one parallel extraction branch."""
    run_id: str
    bid_id: str
    group: str
    fields: list[str]
    feedback: dict[str, str]
    attempt: int
    has_addenda: bool


class QAState(TypedDict, total=False):
    run_id: str
    question: str
    intent: str
    bids: list[str]
    addendum_number: int | None
    queries: list[str]
    evidence: list[Evidence]
    answer: str
    citations: list[Citation]
    answered_by: str
    errors: Annotated[list[str], operator.add]
