"""Cross-encoder re-ranking of fused candidates."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

from fastembed.rerank.cross_encoder import TextCrossEncoder

from settings import ROOT, cfg

if TYPE_CHECKING:
    from search.retrieve import Evidence


@lru_cache(maxsize=1)
def _encoder() -> TextCrossEncoder:
    return TextCrossEncoder(cfg("search", "rerank_model"), cache_dir=str(ROOT / cfg("search", "model_cache")))


def rerank(query: str, evidence: list[Evidence]) -> list[Evidence]:
    """Return evidence sorted by cross-encoder relevance; rerank_score is a 0..1 sigmoid of the logit."""
    if not evidence:
        return []
    logits = list(_encoder().rerank(query, [e.text for e in evidence]))
    scored = [e.model_copy(update={"rerank_score": 1 / (1 + math.exp(-s))})
              for e, s in zip(evidence, logits, strict=True)]
    return sorted(scored, key=lambda e: e.rerank_score or 0.0, reverse=True)
