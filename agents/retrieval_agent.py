"""Retrieval / Search Agent: turns a field (or question) into queries, calls the search engine tool with
metadata filters, re-ranks the merged candidates once, and returns cited evidence.

On a retry it asks the LLM to rewrite the queries using the validator's feedback.
"""
from __future__ import annotations

import time

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import prompt
from agents.schemas import FieldSpec, QueryRewrite
from agents.tracing import Span
from search.rerank import rerank
from search.retrieve import Evidence, SearchFilters, search
from settings import cfg

PER_QUERY = 8


def search_tool(query: str, filters: SearchFilters, span: Span | None, top_k: int = PER_QUERY,
                rerank_results: bool = False) -> list[Evidence]:
    """The search engine exposed as an agent tool; every call is logged on the span."""
    start = time.perf_counter()
    res = search(query, filters=filters, top_k=top_k, rerank=rerank_results)
    if span is not None:
        span.tool("search", {"q": query, **filters.model_dump(exclude_none=True), "top_k": top_k},
                  [f"{e.file_name} p{e.page_number}" for e in res], (time.perf_counter() - start) * 1000)
    return res


def merge_and_rerank(query: str, pools: list[list[Evidence]], keep: int, span: Span | None) -> list[Evidence]:
    merged: dict[str, Evidence] = {}
    for pool in pools:
        for e in pool:
            merged.setdefault(e.chunk_id, e)
    start = time.perf_counter()
    ranked = rerank(query, list(merged.values()))
    if span is not None:
        span.tool("rerank", {"q": query, "candidates": len(merged)},
                  [f"{e.file_name} p{e.page_number} {e.rerank_score:.2f}" for e in ranked[:keep]],
                  (time.perf_counter() - start) * 1000)
    return ranked[:keep]


def rewrite_queries(spec: FieldSpec, feedback: str, llm: LLM, span: Span | None) -> list[str]:
    try:
        out = llm.json(system=prompt("query_rewrite_system"),
                       user=prompt("query_rewrite_user", field=spec.name, description=spec.description,
                                   previous="; ".join(spec.queries), feedback=feedback or "value not found"),
                       schema=QueryRewrite, span=span, small=True, purpose=f"rewrite:{spec.name}")
        return [q for q in out.queries if q.strip()][:4] or spec.queries
    except (LLMError, GuardrailError):
        return spec.queries  # deterministic fallback


def field_evidence(spec: FieldSpec, bid_id: str, has_addenda: bool, span: Span | None, llm: LLM,
                   feedback: str | None = None) -> list[Evidence]:
    queries = spec.queries
    if feedback is not None:  # retry round: rewritten queries + the strongest original one
        queries = list(dict.fromkeys(rewrite_queries(spec, feedback, llm, span) + spec.queries[:1]))
    pools = [search_tool(q, SearchFilters(bid_id=bid_id), span) for q in queries]
    if spec.addendum and has_addenda:  # addenda always in the pool so overrides are visible
        pools.append(search_tool(queries[0], SearchFilters(bid_id=bid_id, doc_types=["addendum"]), span, top_k=4))
    return merge_and_rerank(f"{spec.name}: {queries[0]}", pools, cfg("agents", "evidence_per_field"), span)


def question_evidence(queries: list[str], bid_ids: list[str], span: Span | None, top_k: int,
                      addendum_number: int | None = None) -> list[Evidence]:
    """Q&A retrieval: per bid, all queries, one rerank against the first (user) query."""
    out: list[Evidence] = []
    for bid in bid_ids or [None]:  # type: ignore[list-item]
        filters = SearchFilters(bid_id=bid, addendum_number=addendum_number)
        pools = [search_tool(q, filters, span) for q in queries]
        out += merge_and_rerank(queries[0], pools, top_k, span)
    return out
