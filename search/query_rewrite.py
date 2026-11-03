"""Deterministic query expansion from config/synonyms.yaml (LLM rewriting lives in the Retrieval Agent)."""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from settings import ROOT


@lru_cache(maxsize=1)
def _synonyms() -> dict[str, list[str]]:
    return yaml.safe_load((ROOT / "config" / "synonyms.yaml").read_text()) or {}


def expansions(query: str) -> list[str]:
    """Extra terms for every trigger phrase found in the query (whole-word match), de-duplicated."""
    q = query.lower()
    extra: list[str] = []
    for trigger, terms in _synonyms().items():
        if re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", q):
            extra += [t for t in terms if t.lower() not in q and t not in extra]
    return extra


def expand_query(query: str) -> str:
    extra = expansions(query)
    return f"{query} {' '.join(extra)}" if extra else query
