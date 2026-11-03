"""Extraction-mode agent graph (LangGraph).

    START -> ingest -> plan -> [extract_group x N groups, in parallel] -> reconcile -> validate
                                       ^                                                  |
                                       +---------- failed / not-found fields (retry) -----+
                                                                                          v
                                                                                       report -> END
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.extraction_agents import extract_group
from agents.ingestion_agent import run_ingestion
from agents.llm import LLM, get_llm
from agents.orchestrator import group_tasks, plan_extraction, select_retries
from agents.reconciliation_agent import addenda_numbers, reconcile
from agents.report_agent import build_record, save_outputs
from agents.retrieval_agent import field_evidence
from agents.schemas import BidRecord, load_fields
from agents.state import ExtractState, GroupTask
from agents.tracing import get_tracer, new_tracer
from agents.validator_agent import validate


def build_extract_graph(llm: LLM | None = None) -> Any:
    llm = llm or get_llm()
    specs = load_fields()

    def ingest(state: ExtractState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("ingestion_agent", state["folder"]) as span:
            manifest, errors = run_ingestion(Path(state["folder"]), span, llm)
            span.output({"files": {k: v["doc_type"] for k, v in manifest.items()}, "errors": errors})
        return {"manifest": manifest, "addenda": addenda_numbers(manifest), "errors": errors}

    def plan(state: ExtractState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("orchestrator", {"goal": "extract bid", "bid": state["bid_id"]}) as span:
            steps = plan_extraction(specs, state["manifest"], state["addenda"])
            span.output(steps)
        return {"plan": steps, "pending": list(specs), "feedback": {}, "attempt": 1, "retries": {}}

    def fan_out(state: ExtractState) -> list[Send]:
        return [Send("extract_group", GroupTask(
            run_id=state["run_id"], bid_id=state["bid_id"], group=group, fields=fields,
            feedback={f: state["feedback"][f] for f in fields if f in state["feedback"]},
            attempt=state["attempt"], has_addenda=bool(state["addenda"]),
        )) for group, fields in group_tasks(state["pending"], specs).items()]

    def extract_group_node(task: GroupTask) -> dict[str, Any]:
        tracer = get_tracer(task["run_id"])
        evidence = {}
        with tracer.span("retrieval_agent", {"group": task["group"], "fields": task["fields"],
                                             "attempt": task["attempt"]}) as span:
            for f in task["fields"]:
                evidence[f] = field_evidence(specs[f], task["bid_id"], task["has_addenda"], span, llm,
                                             feedback=task["feedback"].get(f))
            span.output({f: [f"{e.file_name} p{e.page_number}" for e in ev] for f, ev in evidence.items()})
        with tracer.span(f"extraction_agent:{task['group']}", {"fields": task["fields"],
                                                               "feedback": task["feedback"]}) as span:
            drafts = extract_group(task["group"], [specs[f] for f in task["fields"]], evidence, task["bid_id"],
                                   task["feedback"], task["attempt"], span, llm)
            span.output({f: {"status": d.status, "value": d.value} for f, d in drafts.items()})
        chunks = {e.chunk_id: e for ev in evidence.values() for e in ev}
        return {"evidence": evidence, "drafts": drafts, "chunks": chunks}

    def reconcile_node(state: ExtractState) -> dict[str, Any]:
        round_specs = [specs[f] for f in state["pending"]]
        with get_tracer(state["run_id"]).span("reconciliation_agent", {"addenda": state["addenda"],
                                                                       "fields": state["pending"]}) as span:
            drafts, changes, evidence = reconcile(state["bid_id"], state["drafts"], round_specs, state["addenda"],
                                                  span, llm, include_other=state["attempt"] == 1)
            span.output([c.model_dump(include={"field", "addendum_number", "old_value", "new_value"})
                         for c in changes])
        return {"drafts": drafts, "changes": changes, "chunks": {e.chunk_id: e for e in evidence}}

    def validate_node(state: ExtractState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("validator_agent", {"fields": state["pending"],
                                                                  "attempt": state["attempt"]}) as span:
            round_drafts = {f: state["drafts"][f] for f in state["pending"]}
            results = validate(round_drafts, specs, state["chunks"], state["evidence"], span, llm)
            retry, feedback, retries = select_retries(results, state["drafts"], state["retries"])
            span.output({"results": {f: (r.status, r.confidence, r.reasons) for f, r in results.items()},
                         "retry": retry})
        return {"validation": {**state.get("validation", {}), **results}, "pending": retry, "feedback": feedback,
                "retries": retries, "attempt": state["attempt"] + 1}

    def route(state: ExtractState) -> list[Send] | str:
        return fan_out(state) if state["pending"] else "report"

    def report(state: ExtractState) -> dict[str, Any]:
        with get_tracer(state["run_id"]).span("report_agent", {"bid": state["bid_id"]}) as span:
            record = build_record(state["bid_id"], state["run_id"], specs, state["drafts"], state["validation"],
                                  state.get("changes", []), state.get("errors", []), span, llm)
            json_path, md_path = save_outputs(record)
            span.output({"validation": record.validation, "files": [str(json_path), str(md_path)]})
        return {"record": record}

    g = StateGraph(ExtractState)
    g.add_node("ingest", ingest)
    g.add_node("plan", plan)
    g.add_node("extract_group", extract_group_node)  # type: ignore[arg-type]  # receives a Send payload
    g.add_node("reconcile", reconcile_node)
    g.add_node("validate", validate_node)
    g.add_node("report", report)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "plan")
    g.add_conditional_edges("plan", fan_out, ["extract_group"])
    g.add_edge("extract_group", "reconcile")
    g.add_edge("reconcile", "validate")
    g.add_conditional_edges("validate", route, ["extract_group", "report"])
    g.add_edge("report", END)
    return g.compile()


def run_extraction(folder: Path, llm: LLM | None = None) -> tuple[BidRecord, dict[str, Any]]:
    """Extraction mode: bid folder -> BidRecord (also written to outputs/). Returns (record, run summary)."""
    folder = folder.resolve()
    tracer = new_tracer("extract", {"bid_id": folder.name})
    graph = build_extract_graph(llm)
    final = graph.invoke({"run_id": tracer.run_id, "bid_id": folder.name, "folder": str(folder)},
                         {"recursion_limit": 50})
    return final["record"], tracer.close()
