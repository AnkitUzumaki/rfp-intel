"""Question-answering mode (LangGraph): the Orchestrator plans (intent, bids, queries) and routes to
  fact          -> Retrieval Agent + Q&A Agent
  compare       -> Comparison Agent
  addendum_diff -> Reconciliation Agent (addendum passages + saved change log)
then a citation check enforces the guardrail: uncited answers become "Not found in documents."
"""
from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.comparison_agent import compare_bids
from agents.llm import LLM, GuardrailError, LLMError, get_llm
from agents.orchestrator import plan_question
from agents.prompts import EvidenceBook, prompt
from agents.reconciliation_agent import CHANGE_QUERY
from agents.retrieval_agent import question_evidence
from agents.schemas import NOT_FOUND, AnswerResponse, Citation, QAAnswer
from agents.state import QAState
from agents.tracing import Span, get_tracer, new_tracer
from search.index import get_index
from search.retrieve import Evidence
from settings import ROOT, cfg


def answer_from_evidence(question: str, evidence: list[Evidence], span: Span | None, llm: LLM,
                         system: str = "qa_answer_system", context: str = "") -> tuple[str, list[Citation]]:
    if not evidence:
        return f"{NOT_FOUND}.", []
    book = EvidenceBook()
    book.add_all(evidence)
    try:
        resp = llm.json(system=prompt(system), user=prompt("qa_answer_user", question=question, context=context,
                                                           evidence=book.render()),
                        schema=AnswerResponse, span=span, purpose="answer")
    except (LLMError, GuardrailError) as exc:
        return f"Unable to answer right now (LLM unavailable: {exc}).", []
    cites = []
    for eid in book.cited_ids(resp.answer):
        e = book.by_id[eid]
        cites.append(Citation(chunk_id=e.chunk_id, file=e.file_name, page=e.page_number,
                              quote=e.text[:200]))
    return resp.answer.strip(), cites


def _saved_changes(bid: str, number: int | None) -> str:
    path = ROOT / cfg("outputs", "dir") / f"{bid}.json"
    if not path.exists():
        return ""
    changes = json.loads(path.read_text()).get("addendum_changes", [])
    lines = [f"- Addendum {c['addendum_number']} ({c['field']}): {c['description']}" for c in changes
             if number is None or c["addendum_number"] == number]
    return f"Change log from the last extraction run for {bid} (use only if supported by passages):\n" + "\n".join(
        lines) if lines else ""


def build_qa_graph(llm: LLM | None = None) -> Any:
    llm = llm or get_llm()
    top_k = cfg("agents", "qa_top_k")

    def plan(state: QAState) -> dict[str, Any]:
        index = get_index()
        manifests = {b: index.manifest(b) for b in index.bids()}
        with get_tracer(state["run_id"]).span("orchestrator", {"question": state["question"]}) as span:
            p = plan_question(state["question"], manifests, span, llm)
            span.output(p)
        return {"intent": p.intent, "bids": p.bids, "addendum_number": p.addendum_number, "queries": p.queries}

    def route(state: QAState) -> str:
        return {"compare": "compare", "addendum_diff": "addendum_diff"}.get(state["intent"], "fact")

    def fact(state: QAState) -> dict[str, Any]:
        tracer = get_tracer(state["run_id"])
        with tracer.span("retrieval_agent", {"queries": state["queries"], "bids": state["bids"]}) as span:
            evidence = question_evidence(state["queries"], state["bids"], span, top_k)
            span.output([f"{e.bid_id}/{e.file_name} p{e.page_number}" for e in evidence])
        with tracer.span("qa_agent", {"question": state["question"]}) as span:
            answer, cites = answer_from_evidence(state["question"], evidence, span, llm)
            span.output(answer)
        return {"answer": answer, "citations": cites, "answered_by": "qa_agent"}

    def compare(state: QAState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("comparison_agent", {"bids": state["bids"]}) as span:
            answer, cites = compare_bids(state["question"], state["bids"], state["queries"], span, llm)
            span.output(answer)
        return {"answer": answer, "citations": cites, "answered_by": "comparison_agent"}

    def addendum_diff(state: QAState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("reconciliation_agent", {"bids": state["bids"],
                                                                       "addendum": state["addendum_number"]}) as span:
            queries = state["queries"] + [CHANGE_QUERY]
            evidence = question_evidence(queries, state["bids"], span, top_k,
                                         addendum_number=state["addendum_number"])
            evidence = [e for e in evidence if e.doc_type == "addendum"] or evidence
            context = "\n".join(filter(None, (_saved_changes(b, state["addendum_number"]) for b in state["bids"])))
            answer, cites = answer_from_evidence(state["question"], evidence, span, llm, context=context)
            span.output(answer)
        return {"answer": answer, "citations": cites, "answered_by": "reconciliation_agent"}

    def check(state: QAState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("validator_agent", {"answer": state["answer"]}) as span:
            answer, cites = state["answer"], state.get("citations", [])
            if not cites and not answer.startswith(("Not found", "Unable to answer")):
                answer = f"{NOT_FOUND}. (The draft answer had no supporting citation and was withheld.)"
            span.output({"citations": len(cites), "answer": answer})
        return {"answer": answer}

    g = StateGraph(QAState)
    for name, fn in [("plan", plan), ("fact", fact), ("compare", compare), ("addendum_diff", addendum_diff),
                     ("check", check)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route, ["fact", "compare", "addendum_diff"])
    for name in ("fact", "compare", "addendum_diff"):
        g.add_edge(name, "check")
    g.add_edge("check", END)
    return g.compile()


def answer_question(question: str, llm: LLM | None = None) -> QAAnswer:
    tracer = new_tracer("qa", {"question": question})
    final = build_qa_graph(llm).invoke({"run_id": tracer.run_id, "question": question})
    tracer.close()
    return QAAnswer(question=question, answer=final["answer"], citations=final.get("citations", []),
                    answered_by=final.get("answered_by", ""), bids=final.get("bids", []), run_id=tracer.run_id)
