"""Q&A routing, comparison, go/no-go and REST API tests (fake LLM, temporary index)."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agents.qa_agent
import api.app
import search.index
import search.retrieve
from agents.comparison_agent import comparison_report
from agents.go_no_go_agent import evaluate_go_no_go, load_profile
from agents.qa_agent import answer_question
from agents.schemas import BidRecord, FieldOutput, load_fields
from search.index import SearchIndex
from settings import settings
from tests.test_agents import evidence_of, find, make_llm


def planner_and_answer(intent: str, bids: list[str], answer_with: str | None = "due") -> Any:
    def handler(messages: list[dict[str, str]]) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        if "Orchestrator for questions" in system:
            return json.dumps({"intent": intent, "bids": bids, "addendum_number": None, "queries": ["due date"]})
        if "Q&A agent" in system or "Comparison agent" in system:
            ev = evidence_of(user)
            if answer_with is None:
                return json.dumps({"answer": "Proposals are due soon."})  # uncited
            eid = find(ev, answer_with) or next(iter(ev))
            return json.dumps({"answer": f"The answer is in the passage [{eid}]."})
        return json.dumps({})
    return handler


@pytest.fixture
def two_bids(tmp_path: Path, make_pdf, monkeypatch: pytest.MonkeyPatch) -> SearchIndex:
    index = SearchIndex(store_dir=tmp_path / "idx", collection="modes")
    for mod in (search.retrieve, agents.qa_agent, search.index):
        monkeypatch.setattr(mod, "get_index", lambda: index)
    monkeypatch.setitem(settings()["cache"], "enabled", False)
    monkeypatch.setitem(settings()["outputs"], "dir", str(tmp_path / "out"))
    for bid, pages in {"BidA": ["Request for Proposal. Proposals are due March 3, 2025. Warranty: 3 years."],
                       "BidB": ["Request for Proposal. Dell laptops. Warranty: 1 year onsite."]}.items():
        folder = tmp_path / bid
        folder.mkdir()
        shutil.move(make_pdf("RFP.pdf", pages), folder / "RFP.pdf")
        index.index_folder(folder)
    return index


def test_fact_question_routes_to_qa_agent_with_citations(two_bids: SearchIndex) -> None:
    llm, _ = make_llm(planner_and_answer("fact", ["BidA"]))
    a = answer_question("When are proposals due for BidA?", llm)
    assert a.answered_by == "qa_agent" and a.bids == ["BidA"]
    assert a.citations and a.citations[0].file == "RFP.pdf" and a.citations[0].page == 1


def test_compare_question_routes_to_comparison_agent_over_both_bids(two_bids: SearchIndex) -> None:
    llm, clients = make_llm(planner_and_answer("compare", ["BidA", "BidB"], answer_with="warranty"))
    a = answer_question("Compare the warranty of both bids", llm)
    assert a.answered_by == "comparison_agent" and set(a.bids) == {"BidA", "BidB"}
    compare_prompt = next(m[-1]["content"] for m in clients[0].calls if "Comparison" in m[0]["content"])
    assert {"BidA", "BidB"} <= set(re.findall(r"bid=(\w+)", compare_prompt))  # evidence from every bid


def test_addendum_question_routes_to_reconciliation_agent(two_bids: SearchIndex) -> None:
    llm, _ = make_llm(planner_and_answer("addendum_diff", ["BidA"]))
    assert answer_question("What changed in Addendum 1?", llm).answered_by == "reconciliation_agent"


def test_uncited_answer_is_withheld(two_bids: SearchIndex) -> None:
    llm, _ = make_llm(planner_and_answer("fact", ["BidA"], answer_with=None))
    a = answer_question("When are proposals due?", llm)
    assert a.answer.startswith("Not found in documents") and not a.citations


def test_unknown_bid_falls_back_to_all_bids(two_bids: SearchIndex) -> None:
    llm, _ = make_llm(planner_and_answer("fact", ["NoSuchBid"]))
    assert set(answer_question("due date?", llm).bids) == {"BidA", "BidB"}


# ---------- go / no-go + comparison ----------

def _record(bid: str, **values: Any) -> BidRecord:
    fields = {name: FieldOutput() for name in load_fields()} | {"Bid Summary": FieldOutput(value=f"{bid} summary")}
    for k, v in values.items():
        fields[k] = FieldOutput(value=v, sources=[{"file": "f.pdf", "page": 1}], confidence=0.9)
    return BidRecord(bid_id=bid, fields=fields)


def test_go_no_go_decision_follows_profile() -> None:
    rec = _record("BidB", **{"MFG for Registration": ["Dell"], "Due Date": "2024-06-10 14:00 EDT",
                             "Delivery Date": "Within 45 days of award", "Bid Bond Requirement": None})
    profile = load_profile() | {"as_of": "2024-05-25"}
    assert evaluate_go_no_go(rec, profile).decision == "GO"
    assert evaluate_go_no_go(rec, profile | {"authorized_manufacturers": ["HP"]}).decision == "NO-GO"
    late = evaluate_go_no_go(rec, profile | {"as_of": "2024-06-09"})
    assert late.decision == "NO-GO" and any(r.rule == "deadline" and r.status == "fail" for r in late.rules)


def test_comparison_report_side_by_side(tmp_path: Path) -> None:
    text = comparison_report([_record("BidA", Title="Laptops A"), _record("BidB", Title="Laptops B")],
                             tmp_path / "cmp.md")
    assert "| Field | BidA | BidB |" in text and "Laptops A (f.pdf p.1)" in text and "Not found" in text


# ---------- REST API ----------

@pytest.fixture
def client(two_bids: SearchIndex, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(api.app, "DATA_ROOT", tmp_path.resolve())
    return TestClient(api.app.app)


def test_api_health_search_and_envelope(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["success"] and set(body["data"]["bids"]) == {"BidA", "BidB"}
    res = client.get("/search", params={"q": "warranty", "bid_id": "BidB", "top_k": 3}).json()
    assert res["success"] and res["data"]["results"][0]["citation"] == {"file": "RFP.pdf", "page": 1}
    assert all(r["bid_id"] == "BidB" for r in res["data"]["results"])
    missing = client.get("/bids/BidA/record")
    assert missing.status_code == 404 and missing.json()["success"] is False and missing.json()["data"] is None


def test_api_blocks_path_traversal(client: TestClient) -> None:
    for folder in ("../../etc", "/etc", "BidA/../../.."):
        r = client.post("/index", json={"folder": folder})
        assert r.status_code == 400 and not r.json()["success"]
    assert client.get("/bids/..%2F..%2Fsettings/record").status_code == 404


def test_api_validates_input(client: TestClient) -> None:
    assert client.get("/search", params={"q": "x", "top_k": 500}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_rate_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.app, "RATE_LIMIT", 2)
    api.app._hits.clear()
    monkeypatch.setattr(agents.qa_agent, "get_llm", lambda: make_llm(planner_and_answer("fact", ["BidA"]))[0])
    codes = [client.post("/ask", json={"question": "When are proposals due?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
