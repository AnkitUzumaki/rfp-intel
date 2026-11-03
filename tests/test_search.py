"""Search engine tests on an isolated temporary index (real models, small synthetic bids)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from search.cache import SemanticCache
from search.index import SearchIndex
from search.query_rewrite import expand_query, expansions
from search.retrieve import SearchFilters, search


def _bid(root: Path, name: str, files: dict[str, list[str]], make_pdf) -> Path:
    folder = root / name
    folder.mkdir()
    for fname, pages in files.items():
        shutil.move(make_pdf(fname, pages), folder / fname)
    return folder


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> SearchIndex:
    return SearchIndex(store_dir=tmp_path_factory.mktemp("idx"), collection="test")


@pytest.fixture
def bids(tmp_path: Path, make_pdf, index: SearchIndex) -> dict[str, Path]:
    a = _bid(tmp_path, "BidA", {
        "RFP_A.pdf": ["Request for Proposal A-100. Proposals are due March 3, 2025 at 2:00 PM CST.",
                      "Bid bond: a bid security of five percent (5%) is required."],
        "Addendum_1_A.pdf": ["ADDENDUM No. 1. The due date is extended to March 17, 2025 at 2:00 PM CST."],
    }, make_pdf)
    b = _bid(tmp_path, "BidB", {
        "RFP_B.pdf": ["Request for Proposal B-200. Laptops with part number XK-7781-Q. Warranty three years."],
    }, make_pdf)
    for f in (a, b):
        index.index_folder(f)
    return {"BidA": a, "BidB": b}


def test_citations_on_every_result(index: SearchIndex, bids: dict[str, Path]) -> None:
    res = search("bid bond amount", filters=SearchFilters(bid_id="BidA"), use_cache=False, index=index)
    assert res
    top = res[0]
    assert top.file_name == "RFP_A.pdf" and top.page_number == 2 and "bid security" in top.text
    assert all(e.citation()["file"] and e.citation()["page"] >= 1 and e.text for e in res)


def test_bid_filter_isolates_bids(index: SearchIndex, bids: dict[str, Path]) -> None:
    res = search("warranty laptops", filters=SearchFilters(bid_id="BidA"), use_cache=False, index=index)
    assert {e.bid_id for e in res} == {"BidA"}


def test_doc_type_and_addendum_filters(index: SearchIndex, bids: dict[str, Path]) -> None:
    res = search("due date", filters=SearchFilters(bid_id="BidA", doc_types=["addendum"]), use_cache=False,
                 index=index)
    assert res and all(e.doc_type == "addendum" for e in res)
    res = search("due date", filters=SearchFilters(bid_id="BidA", addendum_number=1), use_cache=False, index=index)
    assert res and all(e.addendum_number == 1 for e in res)


def test_keyword_leg_finds_exact_part_number(index: SearchIndex, bids: dict[str, Path]) -> None:
    res = search("XK-7781-Q", mode="bm25", rerank=False, use_cache=False, index=index)
    assert res[0].bid_id == "BidB"


@pytest.mark.parametrize("mode,rerank", [("vector", False), ("bm25", False), ("hybrid", False), ("hybrid", True)])
def test_all_modes_return_results(index: SearchIndex, bids: dict[str, Path], mode: str, rerank: bool) -> None:
    res = search("proposal due date", mode=mode, rerank=rerank, use_cache=False, index=index)  # type: ignore[arg-type]
    assert res
    if rerank:
        scores = [e.rerank_score for e in res]
        assert all(s is not None and 0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True)


def test_incremental_indexing(index: SearchIndex, bids: dict[str, Path], make_pdf) -> None:
    b_count = index.count("BidB")
    again = index.index_folder(bids["BidA"])
    assert again["indexed"] == [] and len(again["skipped"]) == 2
    # change one file in BidA -> only that file re-indexed, BidB untouched
    shutil.move(make_pdf("RFP_A.pdf", ["Request for Proposal A-100 revised. Bid bond not required."]),
                bids["BidA"] / "RFP_A.pdf")
    changed = index.index_folder(bids["BidA"])
    assert changed["indexed"] == ["RFP_A.pdf"] and changed["skipped"] == ["Addendum_1_A.pdf"]
    assert index.count("BidB") == b_count
    res = search("bid bond", filters=SearchFilters(bid_id="BidA", file_name="RFP_A.pdf"), use_cache=False,
                 index=index)
    assert all("five percent" not in e.text for e in res)  # stale chunks deleted
    # removed file -> its chunks deleted
    (bids["BidA"] / "Addendum_1_A.pdf").unlink()
    removed = index.index_folder(bids["BidA"])
    assert removed["removed"] == ["Addendum_1_A.pdf"]
    assert not search("due date", filters=SearchFilters(bid_id="BidA", doc_types=["addendum"]), use_cache=False,
                      index=index)


def test_query_expansion() -> None:
    assert "due date" in expansions("What is the deadline?")
    assert "closing date" in expand_query("deadline")
    assert expand_query("laptop colour") == "laptop colour"


def test_semantic_cache_hit_and_miss(tmp_path: Path) -> None:
    vecs = {"q1": [1.0, 0.0], "q1 again": [0.99, 0.01], "other": [0.0, 1.0]}
    cache = SemanticCache(tmp_path / "c.sqlite", lambda q: np.array(vecs[q]), threshold=0.95)
    cache.put("search", "q1", "k", ["r1"])
    assert cache.get("search", "q1 again", "k") == ["r1"]          # near-duplicate query -> hit
    assert cache.get("search", "other", "k") is None               # different meaning -> miss
    assert cache.get("search", "q1", "different-filters") is None  # same query, other filters -> miss
    assert (cache.hits, cache.misses) == (1, 2)
