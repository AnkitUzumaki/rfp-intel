"""REST API (FastAPI). Every response uses one envelope: {"success": bool, "data": ..., "error": str | null}.

Security: bid folders are resolved inside DATA_ROOT only (no path traversal); LLM-backed endpoints are
rate-limited per client; internal errors are logged server-side and returned as a generic message.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from settings import ROOT, cfg

log = logging.getLogger("api")
DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data")).resolve()
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
_WRITE_LOCK = threading.Lock()  # one index/extract job at a time (embedded Qdrant has a single writer)

app = FastAPI(title="RFP Intelligence Platform", version="1.0",
              description="Hybrid RAG search engine + multi-agent bid extraction and Q&A")


# ---------- envelope + errors ----------

def ok(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None}


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"success": False, "data": None, "error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse({"success": False, "data": None, "error": f"invalid request: {exc.errors()}"},
                        status_code=422)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error: %s", exc)
    return JSONResponse({"success": False, "data": None, "error": "internal error (see server logs)"},
                        status_code=500)


# ---------- guards ----------

_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limited(request: Request) -> None:
    """Sliding one-minute window per client IP for endpoints that call the LLM."""
    # ponytail: in-process limiter; use a shared store (Redis) if the API runs as several replicas
    key = request.client.host if request.client else "unknown"
    now, window = time.monotonic(), _hits[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, f"rate limit: {RATE_LIMIT} requests/minute")
    window.append(now)


def safe_folder(folder: str) -> Path:
    path = (DATA_ROOT / folder).resolve()
    if path != DATA_ROOT and DATA_ROOT not in path.parents:
        raise HTTPException(400, "folder must be inside DATA_ROOT")
    if not path.is_dir():
        raise HTTPException(404, f"bid folder not found: {folder}")
    return path


def record_path(bid_id: str) -> Path:
    out = (ROOT / cfg("outputs", "dir")).resolve()
    path = (out / f"{bid_id}.json").resolve()
    if path.parent != out or not path.exists():
        raise HTTPException(404, f"no extraction output for {bid_id}; POST /extract first")
    return path


# ---------- request models ----------

class FolderRequest(BaseModel):
    folder: str = Field(description="Bid folder name or path relative to DATA_ROOT, e.g. 'Bid1'", min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class CompareRequest(BaseModel):
    bids: list[str] = Field(default_factory=list, description="Bid ids; empty = all extracted bids")
    question: str | None = Field(default=None, max_length=2000, description="Optional aspect to compare")


class BidRequest(BaseModel):
    bid_id: str = Field(min_length=1, max_length=200)


# ---------- endpoints ----------

@app.get("/health")
def health() -> dict[str, Any]:
    from agents.llm import providers_from_env
    from search.index import get_index

    return ok({"status": "ok", "bids": get_index().bids(), "llm_configured": bool(providers_from_env())})


@app.get("/bids")
def bids() -> dict[str, Any]:
    from search.index import get_index

    index = get_index()
    out = {}
    for b in index.bids():
        record = ROOT / cfg("outputs", "dir") / f"{b}.json"
        out[b] = {"files": {f: {k: m.get(k) for k in ("doc_type", "addendum_number", "pages", "errors")}
                            for f, m in index.manifest(b).items()}, "extracted": record.exists()}
    return ok(out)


@app.post("/index")
def index_folder(req: FolderRequest) -> dict[str, Any]:
    from search.index import get_index

    folder = safe_folder(req.folder)
    with _WRITE_LOCK:
        return ok(get_index().index_folder(folder))


@app.get("/search")
def search_endpoint(
    q: str = Query(min_length=1, max_length=1000),
    bid_id: str | None = None,
    doc_type: Literal["bid_page", "rfp", "addendum", "specs", "affidavit", "other"] | None = None,
    addendum_number: int | None = None,
    top_k: int = Query(5, ge=1, le=50),
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
    rerank: bool = True,
) -> dict[str, Any]:
    from search.retrieve import SearchFilters, timed_search

    filters = SearchFilters(bid_id=bid_id, doc_types=[doc_type] if doc_type else None, addendum_number=addendum_number)
    results, ms = timed_search(q, filters=filters, top_k=top_k, mode=mode, rerank=rerank)
    return ok({"query": q, "latency_ms": round(ms, 1), "results": [
        {**e.model_dump(include={"bid_id", "doc_type", "addendum_number", "section", "text", "score",
                                 "rerank_score", "chunk_id"}), "citation": e.citation()} for e in results]})


@app.post("/ask", dependencies=[Depends(rate_limited)])
def ask(req: AskRequest) -> dict[str, Any]:
    from agents.qa_agent import answer_question

    return ok(answer_question(req.question).model_dump())


@app.post("/extract", dependencies=[Depends(rate_limited)])
def extract(req: FolderRequest) -> dict[str, Any]:
    from agents.graph import run_extraction

    folder = safe_folder(req.folder)
    with _WRITE_LOCK:
        record, summary = run_extraction(folder)
    return ok({"record": record.model_dump(), "run": summary})


@app.get("/bids/{bid_id}/record")
def get_record(bid_id: str) -> dict[str, Any]:
    from agents.schemas import BidRecord

    return ok(BidRecord.model_validate_json(record_path(bid_id).read_text()).model_dump())


@app.post("/compare", dependencies=[Depends(rate_limited)])
def compare(req: CompareRequest) -> dict[str, Any]:
    from agents.comparison_agent import comparison_report
    from agents.qa_agent import answer_question
    from agents.schemas import BidRecord

    if req.question:  # free-form aspect -> Comparison Agent via Q&A orchestration
        bids_txt = " and ".join(req.bids) if req.bids else "all bids"
        return ok(answer_question(f"Compare {bids_txt}: {req.question}").model_dump())
    out = ROOT / cfg("outputs", "dir")
    paths = [record_path(b) for b in req.bids] if req.bids else sorted(out.glob("*.json"))
    records = [BidRecord.model_validate_json(p.read_text()) for p in paths]
    if len(records) < 2:
        raise HTTPException(400, "need at least two extracted bids")
    return ok({"markdown": comparison_report(records)})


@app.post("/go-no-go", dependencies=[Depends(rate_limited)])
def go_no_go(req: BidRequest) -> dict[str, Any]:
    from agents.go_no_go_agent import evaluate_go_no_go, save_go_no_go
    from agents.llm import get_llm
    from agents.schemas import BidRecord

    result = evaluate_go_no_go(BidRecord.model_validate_json(record_path(req.bid_id).read_text()), llm=get_llm())
    save_go_no_go(result)
    return ok(result.model_dump())
