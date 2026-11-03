"""Shared fixtures. Failure-case PDFs are generated at test time (no binary fixtures in the repo)."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


def _text_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    return path


@pytest.fixture
def make_pdf(tmp_path: Path):
    def make(name: str, pages: list[str]) -> Path:
        return _text_pdf(tmp_path / name, pages)
    return make


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    return tmp_path / "blank.pdf"


@pytest.fixture
def image_only_pdf(tmp_path: Path) -> Path:
    """A 'scanned' page: text rendered to an image, no text layer."""
    src = pymupdf.open()
    src.new_page().insert_text((72, 100), "BID BOND REQUIRED FIVE PERCENT", fontsize=28)
    png = src[0].get_pixmap(dpi=150).tobytes("png")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=png)
    doc.save(tmp_path / "scan.pdf")
    return tmp_path / "scan.pdf"


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.7\n this is not really a pdf \x00\x01")
    return path


@pytest.fixture
def encrypted_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    return path
