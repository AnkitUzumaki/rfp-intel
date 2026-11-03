"""Unit tests for ingestion: detection, cleaning, classification, PDF/HTML parsing, chunking."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.chunk import _windows, chunk_document, is_heading, split_table
from ingestion.classify import classify, pdf_date
from ingestion.clean import clean_text, remove_headers_footers
from ingestion.detect import detect_file_type
from ingestion.models import Document, Page
from ingestion.parse_html import parse_html
from ingestion.parse_pdf import parse_pdf
from ingestion.pipeline import ingest_file, ingest_folder

# ---------- detect ----------

def test_detect_by_content_not_extension(make_pdf, tmp_path: Path) -> None:
    pdf = make_pdf("a.pdf", ["hello"])
    disguised = tmp_path / "really_a_pdf.html"
    disguised.write_bytes(pdf.read_bytes())
    html = tmp_path / "page.txt"
    html.write_text("<!DOCTYPE html><html><body>x</body></html>")
    junk = tmp_path / "x.pdf"
    junk.write_bytes(b"\x89PNG....")
    assert detect_file_type(pdf) == "pdf"
    assert detect_file_type(disguised) == "pdf"
    assert detect_file_type(html) == "html"
    assert detect_file_type(junk) == "unknown"


# ---------- clean ----------

def test_repeated_headers_footers_removed_except_page_one() -> None:
    body = "\n".join(f"clause {{i}}.{n} body text" for n in range(6))
    pages = [f"ACME RFP 123\nPage {i} of 4\n" + body.format(i=i) + "\nConfidential footer" for i in range(1, 5)]
    out = remove_headers_footers(pages)
    assert "ACME RFP 123" in out[0]
    for text in out[1:]:
        assert "ACME RFP 123" not in text and "Page" not in text and "footer" not in text
        assert "body" in text


def test_short_pages_keep_body_text() -> None:
    pages = [f"Same short body line\nanother body line {i % 2}" for i in range(5)]
    assert remove_headers_footers(pages) == pages  # 2-line pages: no zone, nothing removed


def test_short_documents_keep_headers() -> None:
    pages = ["Title\nbody", "Title\nbody2"]
    assert remove_headers_footers(pages) == pages


def test_clean_text_hyphenation_lines_whitespace() -> None:
    raw = "The require-\nment is   firm.\nVendors shall\nsubmit bids.\n\n\n\nNext   para"
    assert clean_text(raw) == "The requirement is firm.\nVendors shall submit bids.\n\nNext para"


def test_clean_text_keeps_part_number_hyphens() -> None:
    assert "210-\nBLYZ" in clean_text("SKU 210-\nBLYZ")


# ---------- classify ----------

@pytest.mark.parametrize("name,ftype,text,expected", [
    ("Addendum 2 RFP JA-207652 Devices.pdf", "pdf", "", ("addendum", 2)),
    ("Contract_Affidavit.pdf", "pdf", "", ("affidavit", None)),
    ("Dell_Laptop_Specs.pdf", "pdf", "", ("specs", None)),
    ("PORFP_-_Dell_Laptop_Final.pdf", "pdf", "", ("rfp", None)),
    ("bid.html", "html", "", ("bid_page", None)),
    ("JA-207652 Devices FINAL.pdf", "pdf", "PURCHASING DEPARTMENT Request For Proposal 168884", ("rfp", None)),
    ("doc7.pdf", "pdf", "ADDENDUM No. 3\nRFP X", ("addendum", 3)),
    ("Questions and Answers.pdf", "pdf", "Request for Proposal answers", ("other", None)),
    ("misc.pdf", "pdf", "lunch menu", ("other", None)),
])
def test_classify(name: str, ftype: str, text: str, expected: tuple) -> None:
    assert classify(name, ftype, text) == expected


def test_pdf_date() -> None:
    assert pdf_date({"creationDate": "D:20240613175856-05'00'"}) == "2024-06-13"
    assert pdf_date({}) is None


def test_classify_all_real_files(data_dir: Path) -> None:
    expected = {"bid_page": 2, "rfp": 2, "addendum": 2, "specs": 1, "affidavit": 2}
    got: dict[str, int] = {}
    for bid in ("Bid1", "Bid2"):
        for p in (data_dir / bid).iterdir():
            ftype = detect_file_type(p)
            first = parse_pdf(p, ocr_enabled=False)[0][0].text if ftype == "pdf" else ""
            t, _ = classify(p.name, ftype, first)
            got[t] = got.get(t, 0) + 1
    assert got == expected


# ---------- parse_pdf ----------

def test_parse_real_pdf_keeps_pages(data_dir: Path) -> None:
    pages, meta, _ = parse_pdf(data_dir / "Bid1" / "JA-207652 Student and Staff Computing Devices FINAL.pdf",
                               ocr_enabled=False)
    assert len(pages) == 62
    assert [p.page_number for p in pages] == list(range(1, 63))
    assert meta.get("creationDate")


def test_spec_table_extracted_as_markdown(data_dir: Path) -> None:
    pages, _, _ = parse_pdf(data_dir / "Bid2" / "Dell_Laptop_Specs.pdf", ocr_enabled=False)
    tables = [t for p in pages for t in p.tables]
    assert any("210-BLYZ" in t and t.startswith("|") for t in tables)
    # table text is not duplicated into the page text
    assert "210-BLYZ" not in pages[0].text


def test_blank_page_logged_not_crashed(blank_pdf: Path) -> None:
    pages, _, errors = parse_pdf(blank_pdf)
    assert len(pages) == 1 and pages[0].text == ""
    assert errors == ["page 1: empty"]


def test_corrupt_pdf(corrupt_pdf: Path) -> None:
    pages, _, errors = parse_pdf(corrupt_pdf)
    assert pages == [] and errors[0].startswith("unreadable pdf")


def test_encrypted_pdf(encrypted_pdf: Path) -> None:
    pages, _, errors = parse_pdf(encrypted_pdf)
    assert pages == [] and errors == ["encrypted pdf: password required"]


def test_scanned_page_ocr(image_only_pdf: Path) -> None:
    pages, _, errors = parse_pdf(image_only_pdf, ocr_enabled=True)
    assert pages[0].ocr
    assert "BOND" in pages[0].text.upper()
    assert errors == ["page 1: ocr"]


def test_scanned_page_without_ocr_is_reported(image_only_pdf: Path) -> None:
    pages, _, errors = parse_pdf(image_only_pdf, ocr_enabled=False)
    assert pages[0].text == "" and errors == ["page 1: scanned, ocr disabled"]


# ---------- parse_html ----------

def test_bid_page_fields_and_chrome_removed(data_dir: Path) -> None:
    html = next((data_dir / "Bid1").glob("*.html"))
    text, title, fields, errors = parse_html(html)
    assert errors == []
    assert fields["Solicitation Number"] == "JA-207652"
    assert fields["Closing Date"] == "07/09/2024 03:00 PM EDT"
    assert "ProcurementCS@dallasisd.org" in text  # multi-value contact kept
    assert "Closing Date: 07/09/2024 03:00 PM EDT" in text
    assert "## Basic Information" in text
    assert "Saved Searches" not in text  # nav removed
    assert title and "Student and Staff Computing Devices" in title


def test_html_generic_patterns(tmp_path: Path) -> None:
    p = tmp_path / "x.html"
    p.write_text("<html><body><nav>Menu</nav><h2>Info</h2><dl><dt>Bid No</dt><dd>B-1</dd></dl>"
                 "<table><tr><th>Due</th><td>Jan 1</td></tr></table></body></html>")
    text, _, fields, _ = parse_html(p)
    assert fields == {"Bid No": "B-1", "Due": "Jan 1"}
    assert "Menu" not in text and "## Info" in text


# ---------- chunk ----------

def test_headings() -> None:
    assert is_heading("SECTION 3 SCOPE OF WORK")
    assert is_heading("2.1 Delivery Requirements")
    assert is_heading("## Basic Information")
    assert not is_heading("Vendors shall deliver within 30 days of the award.")


def test_windows_overlap() -> None:
    text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(100))
    wins = _windows(text, target=500, overlap=100)
    assert all(len(w) <= 550 for w in wins)
    assert wins[0].splitlines()[-1] in wins[1]  # overlap carries context forward


def test_split_table_repeats_header() -> None:
    md = "| a | b |\n|---|---|\n" + "\n".join(f"| r{i} | v{i} |" for i in range(200))
    parts = split_table(md, 400)
    assert len(parts) > 1
    assert all(p.startswith("| a | b |\n|---|---|") for p in parts)
    assert sum(p.count("| r") for p in parts) == 200  # no row lost or split


def test_chunks_are_page_scoped_and_tables_atomic() -> None:
    body = "\n".join(f"Sentence number {i} about delivery." for i in range(200))
    doc = Document(
        bid_id="B", file_name="f.pdf", file_path="f.pdf", file_type="pdf", content_hash="h", doc_type="rfp",
        pages=[Page(page_number=1, text="1. SCOPE\n" + body, tables=["| h |\n|---|\n| v |"]),
               Page(page_number=2, text="short page")],
    )
    chunks = chunk_document(doc, target=800, overlap=100)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(len(c.text) <= 900 for c in chunks if not c.is_table)
    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1 and tables[0].text == "| h |\n|---|\n| v |"
    assert chunks[0].section == "1. SCOPE"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ---------- pipeline ----------

def test_ingest_folder_bid2(data_dir: Path) -> None:
    docs = ingest_folder(data_dir / "Bid2")
    assert {d.bid_id for d in docs} == {"Bid2"}
    assert sorted(d.doc_type for d in docs) == ["affidavit", "affidavit", "bid_page", "rfp", "specs"]
    assert all(d.content_hash and d.pages for d in docs)


def test_unsupported_file_does_not_crash(tmp_path: Path) -> None:
    p = tmp_path / "notes.docx"
    p.write_bytes(b"PK\x03\x04 zip")
    doc = ingest_file(p, "BidX")
    assert doc.errors == ["unsupported file type"] and doc.pages == []
