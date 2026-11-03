"""Robustness (PDF §10): unseen bid structures, missing fields, empty/odd files, HTML-only bids, long documents."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pymupdf
import pytest

from agents.graph import run_extraction
from search.index import SearchIndex
from search.retrieve import SearchFilters, search
from tests.test_agents import Scripted, make_llm, temp_index  # noqa: F401 (fixture)

BID_PAGE = """<!DOCTYPE html><html><body><nav>Menu Login</nav><main>
<h2>Basic Information</h2><dl><dt>Solicitation Number</dt><dd>XQ-555</dd><dt>Title</dt><dd>Rugged Tablets</dd>
<dt>Closing Date</dt><dd>04/01/2025 02:00 PM EST</dd></dl></main></body></html>"""


@pytest.fixture
def unseen_bid(tmp_path: Path, make_pdf, blank_pdf: Path) -> Path:
    folder = tmp_path / "County RFQ 2025-17"  # spaces, unusual name
    folder.mkdir()
    shutil.move(make_pdf("x1.PDF", ["Request for Quotation XQ-555. Quotes are due April 1, 2025 at 2:00 PM EST."]),
                folder / "solicitation final v3.PDF")
    shutil.move(make_pdf("qa.pdf", ["Questions and Answers. Q: Is delivery required? A: Yes, within 30 days."]),
                folder / "Questions and Answers.pdf")
    shutil.move(blank_pdf, folder / "empty.pdf")
    (folder / "bid page.htm").write_text(BID_PAGE)
    (folder / "readme.docx").write_bytes(b"PK\x03\x04 not supported")
    return folder


def test_unseen_bid_runs_without_code_changes(temp_index: SearchIndex, unseen_bid: Path) -> None:  # noqa: F811
    llm, _ = make_llm(Scripted())
    record, summary = run_extraction(unseen_bid, llm)
    assert record.bid_id == "County RFQ 2025-17" and len(record.fields) == 20
    assert summary["errors"] == []  # no agent step crashed
    errors = " | ".join(record.processing_errors)
    assert "empty.pdf: page 1: empty" in errors and "readme.docx: unsupported file type" in errors
    manifest = temp_index.manifest("County RFQ 2025-17")
    assert manifest["bid page.htm"]["doc_type"] == "bid_page"
    assert manifest["solicitation final v3.PDF"]["doc_type"] == "rfp"
    assert manifest["Questions and Answers.pdf"]["doc_type"] == "other"
    assert manifest["bid page.htm"]["fields"]["Solicitation Number"] == "XQ-555"


def test_html_only_bid(temp_index: SearchIndex, tmp_path: Path) -> None:  # noqa: F811
    folder = tmp_path / "HtmlOnly"
    folder.mkdir()
    (folder / "page.html").write_text(BID_PAGE)
    llm, _ = make_llm(Scripted())
    record, _ = run_extraction(folder, llm)
    assert len(record.fields) == 20 and record.processing_errors == []
    hits = search("solicitation number", filters=SearchFilters(bid_id="HtmlOnly"), index=temp_index, use_cache=False)
    assert hits and "XQ-555" in hits[0].text


def test_long_document_is_indexed_and_searchable(tmp_path: Path) -> None:
    folder = tmp_path / "LongBid"
    folder.mkdir()
    doc = pymupdf.open()
    for i in range(1, 221):
        page = doc.new_page()
        page.insert_text((72, 60), f"City Procurement RFP 99 - General Terms - Page {i} of 220", fontsize=9)
        body = (f"Clause {i}.1 The contractor shall comply with section {i} terms and specifications. " * 4
                if i != 150 else "Bid security: a bid bond of ten percent (10%) of the total bid is required.")
        page.insert_textbox(pymupdf.Rect(72, 90, 540, 760), body, fontsize=10)
    doc.save(folder / "rfp.pdf")
    index = SearchIndex(store_dir=tmp_path / "idx", collection="long")
    start = time.perf_counter()
    stats = index.index_folder(folder)
    elapsed = time.perf_counter() - start
    assert stats["chunks"] >= 220 and elapsed < 300
    hits = search("bid bond percentage", filters=SearchFilters(bid_id="LongBid"), index=index, use_cache=False)
    assert hits[0].page_number == 150
    assert "General Terms - Page" not in hits[0].text  # repeated header removed
