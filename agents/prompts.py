"""Prompt loading (prompts/agents.md) and evidence rendering.

EvidenceBook is the only way document text reaches an LLM prompt: it holds passages returned by the
search tool, gives them short ids (E1, E2, ...), renders them under a size cap and maps the model's
quotes back to citations (file, page, chunk).
"""
from __future__ import annotations

import re
from functools import lru_cache
from string import Template

from agents.schemas import Citation, Quote
from search.retrieve import Evidence
from settings import ROOT, cfg


@lru_cache(maxsize=1)
def _sections() -> dict[str, str]:
    text = (ROOT / "prompts" / "agents.md").read_text()
    parts = re.split(r"^## (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def prompt(name: str, **values: object) -> str:
    return Template(_sections()[name]).substitute({k: str(v) for k, v in values.items()})


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_match(quote: str, text: str) -> float:
    """1.0 if the quote appears verbatim (ignoring case/punctuation/whitespace), else share of quote tokens found."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    tokens = q.split()
    have = set(t.split())
    return sum(tok in have for tok in tokens) / len(tokens)


class EvidenceBook:
    def __init__(self) -> None:
        self.by_id: dict[str, Evidence] = {}
        self._ids: dict[str, str] = {}  # chunk_id -> E-id

    def add(self, ev: Evidence) -> str:
        if ev.chunk_id not in self._ids:
            eid = f"E{len(self._ids) + 1}"
            self._ids[ev.chunk_id] = eid
            self.by_id[eid] = ev
        return self._ids[ev.chunk_id]

    def add_all(self, evidence: list[Evidence]) -> list[str]:
        return [self.add(e) for e in evidence]

    def render(self, max_total: int | None = None, per_passage: int | None = None) -> str:
        max_total = max_total or cfg("agents", "max_prompt_chars")
        per_passage = per_passage or cfg("agents", "evidence_chars")
        out: list[str] = []
        used = 0
        for eid, ev in self.by_id.items():
            kind = ev.doc_type + (f" {ev.addendum_number}" if ev.addendum_number is not None else "")
            body = ev.text if len(ev.text) <= per_passage else ev.text[:per_passage] + " [...]"
            block = f"[{eid}] bid={ev.bid_id} | file={ev.file_name} | page={ev.page_number} | type={kind}\n{body}\n"
            if used + len(block) > max_total:
                out.append(f"[{len(self.by_id) - len(out)} more passages omitted: size limit]")
                break
            out.append(block)
            used += len(block)
        return "\n".join(out) if out else "(no passages retrieved)"

    def citation(self, quote: Quote) -> Citation | None:
        ev = self.by_id.get(quote.evidence_id.strip().strip("[]"))
        if ev is None:
            return None
        return Citation(chunk_id=ev.chunk_id, file=ev.file_name, page=ev.page_number, quote=quote.quote)

    def cited_ids(self, answer: str) -> list[str]:
        return [i for i in dict.fromkeys(re.findall(r"E\d+", answer)) if i in self.by_id]
