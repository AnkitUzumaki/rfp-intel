"""Data contracts for the ingestion pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["bid_page", "rfp", "addendum", "specs", "affidavit", "other"]


class Page(BaseModel):
    page_number: int
    text: str = ""
    tables: list[str] = Field(default_factory=list)  # markdown, one per table
    ocr: bool = False


class Document(BaseModel):
    bid_id: str
    file_name: str
    file_path: str
    file_type: Literal["pdf", "html", "unknown"]
    content_hash: str
    doc_type: DocType = "other"
    addendum_number: int | None = None
    doc_date: str | None = None  # ISO date
    title: str | None = None
    pages: list[Page] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)  # label -> value pairs (HTML bid pages)
    errors: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: DocType
    addendum_number: int | None
    doc_date: str | None
    page_number: int
    section: str | None
    is_table: bool
    text: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedder: a short context header improves recall for terse chunks."""
        head = f"{self.doc_type}"
        if self.addendum_number is not None:
            head += f" {self.addendum_number}"
        if self.section:
            head += f" | {self.section}"
        return f"[{head}]\n{self.text}"
