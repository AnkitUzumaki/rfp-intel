"""Hybrid retrieval: dense + BM25 fused with RRF, metadata filters, optional cross-encoder rerank.

Every result carries its citation (file, page, chunk text).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict
from qdrant_client import models

from ingestion.models import DocType
from search.cache import get_cache
from search.index import DENSE, SPARSE, SearchIndex, get_index
from search.query_rewrite import expand_query
from search.rerank import rerank as cross_rerank
from settings import cfg

Mode = Literal["vector", "bm25", "hybrid"]
# ponytail: one lock around embedded-Qdrant queries (ms each); parallel agents still overlap on LLM/rerank time
_QDRANT_LOCK = threading.Lock()


class SearchFilters(BaseModel):
    bid_id: str | None = None
    doc_types: list[DocType] | None = None
    addendum_number: int | None = None
    file_name: str | None = None

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.bid_id:
            must.append(models.FieldCondition(key="bid_id", match=models.MatchValue(value=self.bid_id)))
        if self.doc_types:
            types = [str(t) for t in self.doc_types]
            must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=types)))
        if self.addendum_number is not None:
            must.append(models.FieldCondition(key="addendum_number",
                                              match=models.MatchValue(value=self.addendum_number)))
        if self.file_name:
            must.append(models.FieldCondition(key="file_name", match=models.MatchValue(value=self.file_name)))
        return models.Filter(must=must) if must else None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    bid_id: str
    file_name: str
    page_number: int
    doc_type: DocType
    addendum_number: int | None = None
    doc_date: str | None = None
    section: str | None = None
    is_table: bool = False
    text: str
    score: float = 0.0
    rerank_score: float | None = None

    def citation(self) -> dict[str, str | int]:
        return {"file": self.file_name, "page": self.page_number}


def _version(index: SearchIndex) -> str:
    """Changes whenever any indexed file changes -> stale cache entries never match."""
    state = {b: {f: m["content_hash"] for f, m in index.manifest(b).items()} for b in index.bids()}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


def _query_points(index: SearchIndex, query: str, flt: models.Filter | None, mode: Mode, limit: int,
                  expand: bool) -> list[Evidence]:
    dense = next(iter(index.dense.query_embed(query))).tolist()
    s = next(iter(index.sparse.query_embed(expand_query(query) if expand else query)))
    sparse = models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    with _QDRANT_LOCK:
        res = _run_query(index, dense, sparse, flt, mode, limit)
    return [Evidence(**(p.payload or {}), score=p.score) for p in res.points]


def _run_query(index: SearchIndex, dense: list[float], sparse: models.SparseVector, flt: models.Filter | None,
               mode: Mode, limit: int) -> models.QueryResponse:
    if mode == "vector":
        return index.client.query_points(index.collection, query=dense, using=DENSE, query_filter=flt, limit=limit)
    if mode == "bm25":
        return index.client.query_points(index.collection, query=sparse, using=SPARSE, query_filter=flt, limit=limit)
    n = cfg("search", "candidates")
    return index.client.query_points(
        index.collection,
        prefetch=[models.Prefetch(query=dense, using=DENSE, filter=flt, limit=n),
                  models.Prefetch(query=sparse, using=SPARSE, filter=flt, limit=n)],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )


def search(
    query: str,
    filters: SearchFilters | None = None,
    top_k: int | None = None,
    mode: Mode = "hybrid",
    rerank: bool = True,
    expand: bool | None = None,
    use_cache: bool | None = None,
    index: SearchIndex | None = None,
) -> list[Evidence]:
    index = index or get_index()
    top_k = top_k or cfg("search", "top_k")
    expand = cfg("search", "expand_queries") if expand is None else expand
    use_cache = cfg("cache", "enabled") if use_cache is None else use_cache
    filters = filters or SearchFilters()
    key = json.dumps([filters.model_dump(), mode, rerank, expand, top_k, _version(index)], sort_keys=True)
    cache = get_cache(index) if use_cache else None
    if cache and (hit := cache.get("search", query, key)) is not None:
        return [Evidence(**e) for e in hit]

    limit = cfg("search", "rerank_candidates") if rerank else top_k
    results = _query_points(index, query, filters.to_qdrant(), mode, max(limit, top_k), expand)
    if rerank:
        results = cross_rerank(query, results)
    results = results[:top_k]
    if cache:
        cache.put("search", query, key, [e.model_dump() for e in results])
    return results


def timed_search(query: str, **kwargs: object) -> tuple[list[Evidence], float]:
    start = time.perf_counter()
    res = search(query, **kwargs)  # type: ignore[arg-type]
    return res, (time.perf_counter() - start) * 1000
