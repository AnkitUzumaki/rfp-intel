"""Search index: Qdrant collection with dense (bge) + sparse (BM25) vectors and chunk metadata payload.

Incremental: a file is re-ingested only when its content hash changed; its old chunks are replaced.
Files removed from a bid folder have their chunks deleted. Other bids are never touched.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from ingestion.models import Chunk, Document
from ingestion.pipeline import PIPELINE_VERSION, chunk_documents, ingest_file
from settings import ROOT, cfg

log = logging.getLogger(__name__)

DENSE, SPARSE = "dense", "bm25"
_KEYWORD_FIELDS = ("bid_id", "doc_type", "file_name")


class SearchIndex:
    def __init__(self, store_dir: Path | None = None, url: str | None = None, collection: str | None = None):
        self.store = Path(store_dir or ROOT / cfg("search", "store_dir"))
        self.store.mkdir(parents=True, exist_ok=True)
        url = url if url is not None else os.getenv("QDRANT_URL") or None
        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(self.store / "qdrant"))
        self.collection = collection or cfg("search", "collection")
        cache_dir = str(ROOT / cfg("search", "model_cache"))
        self.dense = TextEmbedding(cfg("search", "dense_model"), cache_dir=cache_dir)
        self.sparse = SparseTextEmbedding(cfg("search", "sparse_model"), cache_dir=cache_dir)
        self._ensure_collection()

    # ---------- setup ----------
    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        dim = len(next(iter(self.dense.embed(["probe"]))))
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in _KEYWORD_FIELDS:
            self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "addendum_number", models.PayloadSchemaType.INTEGER)

    # ---------- manifests: per-bid file registry ----------
    def _manifest_path(self, bid_id: str) -> Path:
        return self.store / "manifests" / f"{bid_id}.json"

    def manifest(self, bid_id: str) -> dict[str, dict[str, Any]]:
        path = self._manifest_path(bid_id)
        return json.loads(path.read_text()) if path.exists() else {}

    def bids(self) -> list[str]:
        folder = self.store / "manifests"
        return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []

    def _save_manifest(self, bid_id: str, data: dict[str, dict[str, Any]]) -> None:
        path = self._manifest_path(bid_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    # ---------- writes ----------
    def _delete_file(self, bid_id: str, file_name: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="bid_id", match=models.MatchValue(value=bid_id)),
            models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name)),
        ])))

    def upsert_chunks(self, chunks: list[Chunk], content_hash: str) -> None:
        if not chunks:
            return
        texts = [c.embed_text for c in chunks]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d.tolist(),
                        SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                payload={**c.model_dump(), "content_hash": content_hash},
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def index_document(self, doc: Document) -> int:
        self._delete_file(doc.bid_id, doc.file_name)
        chunks = chunk_documents([doc])
        self.upsert_chunks(chunks, doc.content_hash)
        return len(chunks)

    def index_folder(self, folder: Path) -> dict[str, Any]:
        """Index one bid folder incrementally. Returns stats incl. per-file processing errors."""
        bid_id = folder.name
        manifest = self.manifest(bid_id)
        files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
        stats: dict[str, Any] = {"bid_id": bid_id, "indexed": [], "skipped": [], "removed": [], "chunks": 0}
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            known = manifest.get(path.name, {})
            if known.get("content_hash") == digest and known.get("pipeline") == PIPELINE_VERSION:
                stats["skipped"].append(path.name)
                continue
            doc = ingest_file(path, bid_id)
            stats["chunks"] += self.index_document(doc)
            stats["indexed"].append(path.name)
            manifest[path.name] = {
                "content_hash": doc.content_hash, "pipeline": PIPELINE_VERSION, "doc_type": doc.doc_type,
                "addendum_number": doc.addendum_number,
                "doc_date": doc.doc_date, "title": doc.title, "pages": len(doc.pages), "errors": doc.errors,
                "fields": doc.fields,
            }
        for gone in set(manifest) - {p.name for p in files}:
            self._delete_file(bid_id, gone)
            manifest.pop(gone)
            stats["removed"].append(gone)
        self._save_manifest(bid_id, manifest)
        log.info("indexed %s: %s", bid_id, {k: v for k, v in stats.items() if k != "bid_id"})
        return stats

    def set_doc_meta(self, bid_id: str, file_name: str, doc_type: str, addendum_number: int | None) -> None:
        """Reclassify an indexed file (e.g. LLM fallback for 'other' docs) without re-embedding."""
        flt = models.Filter(must=[
            models.FieldCondition(key="bid_id", match=models.MatchValue(value=bid_id)),
            models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name)),
        ])
        self.client.set_payload(self.collection, payload={"doc_type": doc_type, "addendum_number": addendum_number},
                                points=models.FilterSelector(filter=flt))
        manifest = self.manifest(bid_id)
        manifest[file_name].update(doc_type=doc_type, addendum_number=addendum_number, reclassified=True)
        self._save_manifest(bid_id, manifest)

    def count(self, bid_id: str | None = None) -> int:
        flt = models.Filter(must=[models.FieldCondition(key="bid_id", match=models.MatchValue(value=bid_id))]) \
            if bid_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    """Process-wide index (embedded Qdrant allows one client per storage path)."""
    index = SearchIndex()
    atexit.register(index.client.close)  # release the storage lock cleanly before interpreter teardown
    return index
