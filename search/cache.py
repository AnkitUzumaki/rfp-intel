"""Semantic cache: reuse a stored result when a new query is near-identical (cosine >= threshold)
and was asked with the same key (filters, mode, index version)."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from settings import cfg

if TYPE_CHECKING:
    from search.index import SearchIndex

Embed = Callable[[str], np.ndarray]


class SemanticCache:
    def __init__(self, path: Path, embed: Embed, threshold: float):
        self.embed, self.threshold = embed, threshold
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (ns TEXT, key TEXT, query TEXT, emb BLOB, result TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix ON cache (ns, key)")
        self.hits = self.misses = 0

    def _vec(self, query: str) -> np.ndarray:
        v = np.asarray(self.embed(query), dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def get(self, ns: str, query: str, key: str) -> Any | None:
        with self._lock:
            rows = self.db.execute("SELECT emb, result FROM cache WHERE ns=? AND key=?", (ns, key)).fetchall()
        if rows:
            q = self._vec(query)
            sims = [float(np.frombuffer(emb, dtype=np.float32) @ q) for emb, _ in rows]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                self.hits += 1
                return json.loads(rows[best][1])
        self.misses += 1
        return None

    def put(self, ns: str, query: str, key: str, result: Any) -> None:
        with self._lock:
            self.db.execute("INSERT INTO cache VALUES (?,?,?,?,?)",
                            (ns, key, query, self._vec(query).tobytes(), json.dumps(result)))
            self.db.commit()


@lru_cache(maxsize=1)
def get_cache(index: SearchIndex) -> SemanticCache:
    return SemanticCache(
        index.store / "cache.sqlite",
        lambda q: next(iter(index.dense.query_embed(q))),
        cfg("cache", "threshold"),
    )
