"""Agent system tests with a scripted fake LLM (no network) on an isolated temporary index.

Covers: end-to-end extraction graph, addendum reconciliation, validator feedback loop (retry then pass),
max-retry stop, the four failure fixtures (timeout, malformed JSON, empty retrieval, 429), the
search-as-tool guardrail, tracing, and the §8.1 output shape.
"""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

import agents.ingestion_agent
import search.retrieve
from agents.extraction_agents import extract_group
from agents.graph import run_extraction
from agents.llm import LLM, GuardrailError, LLMError, Provider
from agents.schemas import BidRecord, ExtractionResponse, load_fields
from agents.tracing import Tracer
from search.index import SearchIndex
from settings import settings

Handler = Callable[[list[dict[str, str]]], str]
EVIDENCE = re.compile(r"\[(E\d+)\] bid=(\S+) \| file=(.+?) \| page=(\d+) \| type=([^\n]+)\n(.*?)"
                      r"(?=\n\n\[E\d+\] |\n\nReturn one entry|\s*\Z)", re.S)


# ---------- fakes ----------

class FakeClient:
    def __init__(self, handler: Handler):
        self.handler, self.calls = handler, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model: str, messages: list[dict[str, str]], **_: Any) -> Any:
        self.calls.append(messages)
        content = self.handler(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def make_llm(*handlers: Handler) -> tuple[LLM, list[FakeClient]]:
    clients = [FakeClient(h) for h in handlers]
    providers = [Provider(f"p{i}", "", f"model{i}", f"small{i}", "k") for i in range(len(handlers))]
    by_name = {p.name: c for p, c in zip(providers, clients, strict=True)}
    names = iter(by_name)
    return LLM(providers, client_factory=lambda **_: by_name[next(names)]), clients


def evidence_of(user: str) -> dict[str, tuple[str, str]]:
    return {m[0]: (m[2], m[5]) for m in EVIDENCE.findall(user)}


def find(ev: dict[str, tuple[str, str]], needle: str) -> str | None:
    return next((eid for eid, (_, text) in ev.items() if needle.lower() in text.lower()), None)


class Scripted:
    """Answers each agent's prompt like a well-behaved model would, using only the given evidence."""

    def __init__(self, due_first: str = "2025-03-03 14:00 CST", always_bad_quote: bool = False):
        self.due_first, self.always_bad_quote = due_first, always_bad_quote

    def __call__(self, messages: list[dict[str, str]]) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        if "extraction specialist" in system:
            return self.extract(user)
        if "Addendum Reconciliation" in system:
            ev = evidence_of(user)
            eid = find(ev, "extended to March 17")
            changes = [{"field": "Due Date", "addendum_number": 1, "old_value": "2025-03-03 14:00 CST",
                        "new_value": "2025-03-17 14:00 CST", "description": "Due date extended to March 17.",
                        "quote": {"evidence_id": eid, "quote": "The due date is extended to March 17, 2025"}}
                       ] if eid else []
            return json.dumps({"changes": changes})
        if "You are the Validator" in system:
            fields = re.findall(r"- field: (.+)", user)
            return json.dumps({"verdicts": [{"field": f, "supported": True} for f in fields]})
        if "search queries" in system:
            return json.dumps({"queries": ["proposal due date", "submission deadline"]})
        if "Report agent" in system:
            return json.dumps({"summary": "Bid A buys laptops. Proposals are due March 17, 2025. A 5% bid bond "
                                          "is required.", "fields_used": ["Due Date", "Bid Bond Requirement"]})
        return json.dumps({})

    def extract(self, user: str) -> str:
        ev = evidence_of(user)
        fields = re.findall(r"^- (.+?): ", user, re.M)
        retry = "previous attempt was rejected" in user
        out = []
        for f in fields:
            item: dict[str, Any] = {"field": f, "found": False, "value": None, "quotes": []}
            if f == "Due Date" and (eid := find(ev, "Proposals are due")):
                value = self.due_first if not retry else "2025-03-03 14:00 CST"
                quote = "zebra unicorn" if self.always_bad_quote else "Proposals are due March 3, 2025 at 2:00 PM CST"
                item = {"field": f, "found": True, "value": value, "quotes": [{"evidence_id": eid, "quote": quote}]}
            if f == "Bid Bond Requirement" and (eid := find(ev, "bid security")):
                item = {"field": f, "found": True, "value": "5% bid security",
                        "quotes": [{"evidence_id": eid, "quote": "a bid security of five percent (5%) is required"}]}
            out.append(item)
        return json.dumps({"fields": out})


def trace_steps(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines = [json.loads(line) for line in next((out_dir / "traces").glob("*.jsonl")).read_text().splitlines()]
    return lines[:-1], lines[-1]


# ---------- fixtures ----------

@pytest.fixture
def temp_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SearchIndex:
    index = SearchIndex(store_dir=tmp_path / "idx", collection="agents")
    monkeypatch.setattr(search.retrieve, "get_index", lambda: index)
    monkeypatch.setattr(agents.ingestion_agent, "get_index", lambda: index)
    monkeypatch.setitem(settings()["outputs"], "dir", str(tmp_path / "out"))
    monkeypatch.setitem(settings()["llm"], "backoff_s", 0)
    monkeypatch.setitem(settings()["cache"], "enabled", False)
    return index


@pytest.fixture
def bid_folder(tmp_path: Path, make_pdf) -> Path:
    folder = tmp_path / "BidA"
    folder.mkdir()
    files = {
        "RFP_A.pdf": ["Request for Proposal A-100 for student laptops. Proposals are due March 3, 2025 at 2:00 PM CST.",
                      "Bid bond: a bid security of five percent (5%) is required with the proposal."],
        "Addendum_1_A.pdf": ["ADDENDUM No. 1. The due date is extended to March 17, 2025 at 2:00 PM CST."],
    }
    for name, pages in files.items():
        shutil.move(make_pdf(name, pages), folder / name)
    return folder


# ---------- end-to-end extraction ----------

def test_extraction_graph_end_to_end(temp_index: SearchIndex, bid_folder: Path) -> None:
    llm, _ = make_llm(Scripted())
    record, summary = run_extraction(bid_folder, llm)

    assert isinstance(record, BidRecord) and record.bid_id == "BidA"
    assert set(record.fields) == set(load_fields()) | {"Bid Summary"}  # all 20 fields
    due = record.fields["Due Date"]
    assert due.value == "2025-03-17 14:00 CST"                         # addendum override applied
    assert {"file": "Addendum_1_A.pdf", "page": 1} in due.sources
    assert "Changed by Addendum 1" in due.notes and "2025-03-03" in due.notes
    assert record.fields["Bid Bond Requirement"].value == "5% bid security"
    assert record.addendum_changes and record.addendum_changes[0].field == "Due Date"
    title = record.fields["Title"]
    assert title.value is None and title.sources == [] and "Not found in documents" in title.notes
    assert sum(record.validation.values()) == 20
    assert all(f.sources for f in record.fields.values() if f.value is not None)  # every value cited

    out = Path(settings()["outputs"]["dir"])
    saved = json.loads((out / "BidA.json").read_text())
    assert {"bid_id", "fields", "addendum_changes", "validation", "processing_errors"} <= set(saved)
    assert "## Fields" in (out / "BidA.md").read_text()
    steps, footer = trace_steps(out)
    agents_seen = {s["agent"].split(":")[0] for s in steps}
    assert {"ingestion_agent", "orchestrator", "retrieval_agent", "extraction_agent", "reconciliation_agent",
            "validator_agent", "report_agent"} <= agents_seen
    assert all({"input", "tool_calls", "output", "tokens", "latency_ms"} <= set(s) for s in steps)
    assert footer["type"] == "run_summary" and footer["llm_calls"] > 0 and footer["prompt_tokens"] > 0
    assert summary["errors"] == []
    first_round = [s for s in steps if s["agent"].startswith("extraction_agent") and s["input"]["feedback"] == {}]
    assert len(first_round) == 4  # 4 field groups dispatched in parallel


def test_guardrail_prompts_contain_only_retrieved_evidence(temp_index: SearchIndex, bid_folder: Path) -> None:
    llm, clients = make_llm(Scripted())
    run_extraction(bid_folder, llm)
    indexed = {p.payload["text"] for p in temp_index.client.scroll(temp_index.collection, limit=1000)[0]}
    cap = settings()["agents"]["max_prompt_chars"]
    for messages in clients[0].calls:
        user = messages[-1]["content"]
        assert len(user) <= cap + 8000
        for _, (_, text) in evidence_of(user).items():
            body = text.strip().removesuffix("[...]").strip()
            assert any(body in chunk for chunk in indexed), "prompt contains text that is not a retrieved chunk"
    with pytest.raises(GuardrailError):
        llm.json(system="s", user="x" * (cap + 10_000), schema=ExtractionResponse)


# ---------- feedback loop ----------

def test_validator_rejection_triggers_retry_then_passes(temp_index: SearchIndex, bid_folder: Path) -> None:
    (bid_folder / "Addendum_1_A.pdf").unlink()  # isolate the loop from reconciliation
    llm, _ = make_llm(Scripted(due_first="March 3, 2025"))  # wrong format on the first attempt
    record, _ = run_extraction(bid_folder, llm)
    assert record.fields["Due Date"].value == "2025-03-03 14:00 CST"
    steps, _ = trace_steps(Path(settings()["outputs"]["dir"]))
    retried = [s for s in steps if s["agent"] == "extraction_agent:dates_logistics" and s["input"]["feedback"]]
    assert retried and "format" in json.dumps(retried[0]["input"]["feedback"])


def test_max_retries_stops_and_never_publishes_unsupported(temp_index: SearchIndex, bid_folder: Path) -> None:
    (bid_folder / "Addendum_1_A.pdf").unlink()
    llm, _ = make_llm(Scripted(always_bad_quote=True))
    record, _ = run_extraction(bid_folder, llm)
    due = record.fields["Due Date"]
    assert due.value is None and "unverified candidate rejected" in due.notes
    steps, _ = trace_steps(Path(settings()["outputs"]["dir"]))
    rounds = [s for s in steps if s["agent"] == "extraction_agent:dates_logistics"]
    assert len(rounds) == settings()["agents"]["max_retries"] + 1  # first attempt + max retries, then stop


# ---------- failure fixtures (R51) ----------

def _req() -> httpx.Request:
    return httpx.Request("POST", "http://llm/v1/chat/completions")


def test_timeout_falls_back_to_second_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(settings()["llm"], "backoff_s", 0)

    def timeout(_: Any) -> str:
        raise openai.APITimeoutError(request=_req())
    llm, _ = make_llm(timeout, lambda _: json.dumps({"fields": []}))
    with Tracer("test", out_dir=tmp_path).span("x") as span:
        out = llm.json(system="s", user="u", schema=ExtractionResponse, span=span)
    assert out.fields == []
    calls = span.record["llm_calls"]
    assert [c["ok"] for c in calls] == [False] * settings()["llm"]["max_attempts"] + [True]
    assert calls[-1]["provider"] == "p1"


def test_rate_limit_retries_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(settings()["llm"], "backoff_s", 0)
    state = {"n": 0}

    def flaky(_: Any) -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise openai.RateLimitError("429", response=httpx.Response(429, request=_req()), body=None)
        return json.dumps({"fields": []})
    llm, _ = make_llm(flaky)
    assert llm.json(system="s", user="u", schema=ExtractionResponse).fields == []
    assert state["n"] == 2


def test_malformed_json_is_repaired_once_then_fails() -> None:
    replies = iter(["not json at all", '```json\n{"fields": []}\n```'])
    llm, clients = make_llm(lambda _: next(replies))
    assert llm.json(system="s", user="u", schema=ExtractionResponse).fields == []
    assert "not valid for the schema" in clients[0].calls[1][-1]["content"]  # repair turn sent

    llm_bad, _ = make_llm(lambda _: "still not json")
    with pytest.raises(LLMError):
        llm_bad.json(system="s", user="u", schema=ExtractionResponse)


def test_empty_retrieval_marks_not_found_without_llm_call() -> None:
    llm, clients = make_llm(lambda _: pytest.fail("LLM must not be called without evidence"))
    specs = load_fields()
    drafts = extract_group("org_contact", [specs["Title"]], {"Title": []}, "BidX", {}, 1, None, llm)
    assert drafts["Title"].status == "not_found" and drafts["Title"].notes == "Not found in documents"
    assert clients[0].calls == []


def test_no_provider_configured_raises_clean_error() -> None:
    with pytest.raises(LLMError, match="no LLM provider"):
        LLM(providers=[]).json(system="s", user="u", schema=ExtractionResponse)
