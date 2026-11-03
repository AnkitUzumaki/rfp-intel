"""Validator rule checks: formats per field type, cross-field consistency, hallucination, confidence."""
from __future__ import annotations

import pytest

from agents.llm import LLM
from agents.prompts import quote_match
from agents.schemas import Citation, FieldDraft, load_fields
from agents.validator_agent import consistency_problems, format_problems, is_unsupported, parse_date, validate
from search.retrieve import Evidence

SPECS = load_fields()


@pytest.mark.parametrize("field,value,ok", [
    ("Due Date", "2024-07-09 14:00 CST", True),
    ("Due Date", "July 9, 2024 2 PM", False),
    ("Delivery Date", "Within 45 days of award", True),
    ("Delivery Date", "ASAP", False),
    ("Bid Submission Type", ["electronic_portal", "sealed_paper"], True),
    ("Bid Submission Type", ["carrier pigeon"], False),
    ("Model_no", ["Latitude 5550"], True),
    ("Model_no", "Latitude 5550", False),
    ("contact_info", {"name": "A", "email": "a@b.org", "phone": "1"}, True),
    ("contact_info", {"name": "A", "email": "not-an-email"}, False),
    ("contact_info", {"name": "A", "fax": "1"}, False),
    ("Pre Bid Meeting", "None", True),            # explicit negative allowed (negative: true)
    ("Bid Bond Requirement", "Not required", True),
    ("Title", "  ", False),
])
def test_format_rules(field: str, value: object, ok: bool) -> None:
    assert (format_problems(SPECS[field], value) == []) is ok


def test_parse_date_formats() -> None:
    assert parse_date("2024-07-09 14:00 CST").day == 9  # type: ignore[union-attr]
    assert parse_date("06/10/2024 02:00 PM EDT").month == 6  # type: ignore[union-attr]
    assert parse_date("July 9, 2024").month == 7  # type: ignore[union-attr]
    assert parse_date("10-JUN-2024 14:00:00").day == 10  # type: ignore[union-attr]
    assert parse_date("sometime soon") is None


def test_consistency_and_hallucination() -> None:
    drafts = {
        "Due Date": FieldDraft(field="Due Date", value="2024-06-10 14:00 EDT", status="found"),
        "Pre Bid Meeting": FieldDraft(field="Pre Bid Meeting", value={"date": "2024-06-20"}, status="found"),
        "Part_no": FieldDraft(field="Part_no", value=["210-BLYZ", "999-FAKE"], status="found"),
    }
    problems = consistency_problems(drafts, "SKU 210-BLYZ base")
    assert "after the due date" in problems["Pre Bid Meeting"][0]
    assert "999-FAKE" in problems["Part_no"][0] and "210-BLYZ'" not in problems["Part_no"][0]


def test_quote_match() -> None:
    text = "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST."
    assert quote_match("new due date for this RFP will be July 9, 2024", text) == 1.0
    assert quote_match("NEW DUE DATE  for this rfp", text) == 1.0
    assert quote_match("bid bond five percent", text) < 0.5


def test_validate_scores_and_flags_unsupported() -> None:
    ev = Evidence(chunk_id="c1", bid_id="B", file_name="a.pdf", page_number=1, doc_type="addendum",
                  text="The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.", rerank_score=0.9)
    good = FieldDraft(field="Due Date", value="2024-07-09 14:00 CST", status="found",
                      citations=[Citation(chunk_id="c1", file="a.pdf", page=1, quote="July 9, 2024 at 2:00 PM CST")])
    bad = FieldDraft(field="Title", value="Invented title", status="found",
                     citations=[Citation(chunk_id="c1", file="a.pdf", page=1, quote="laptops for students")])
    missing = FieldDraft(field="Payment Terms", status="not_found", notes="Not found in documents")
    res = validate({"Due Date": good, "Title": bad, "Payment Terms": missing}, SPECS, {"c1": ev},
                   {"Due Date": [ev], "Payment Terms": []}, None, LLM(providers=[]))  # judge unavailable -> neutral
    assert res["Due Date"].status == "passed" and res["Due Date"].confidence >= 0.7
    assert res["Title"].status == "failed" and is_unsupported(res["Title"])
    assert res["Payment Terms"].status == "not_found" and res["Payment Terms"].confidence == 1.0
