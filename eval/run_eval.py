"""Retrieval evaluation: Recall@k, MRR, nDCG@10 and latency per search configuration.

Relevance is page-level (file + page), so the gold set stays valid when chunking changes.
Usage:
    python -m eval.run_eval                      # all configs on the main index
    python -m eval.run_eval --chunk-sizes 1000,2000,3000   # also compare chunk sizes (separate indexes)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from search.index import SearchIndex, get_index
from search.retrieve import SearchFilters, timed_search
from settings import ROOT, settings

GOLD = ROOT / "eval" / "gold.jsonl"
K = 10

CONFIGS: dict[str, dict[str, Any]] = {
    "vector-only": {"mode": "vector", "rerank": False, "expand": False},
    "bm25-only": {"mode": "bm25", "rerank": False, "expand": False},
    "bm25+expand": {"mode": "bm25", "rerank": False, "expand": True},
    "hybrid": {"mode": "hybrid", "rerank": False, "expand": False},
    "hybrid+expand": {"mode": "hybrid", "rerank": False, "expand": True},
    "hybrid+rerank": {"mode": "hybrid", "rerank": True, "expand": False},
    "hybrid+expand+rerank": {"mode": "hybrid", "rerank": True, "expand": True},
}


def load_gold(path: Path = GOLD) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_ranking(ranked: list[tuple[str, int]], relevant: set[tuple[str, int]]) -> dict[str, float]:
    """Binary page-level relevance; repeated chunks from the same page count once."""
    seen: set[tuple[str, int]] = set()
    gains: list[int] = []
    for hit in ranked[:K]:
        gains.append(1 if hit in relevant and hit not in seen else 0)
        seen.add(hit)
    first = next((i for i, g in enumerate(gains) if g), None)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), K)))
    return {
        "recall@1": float(first is not None and first < 1),
        "recall@5": float(first is not None and first < 5),
        "recall@10": float(first is not None),
        "mrr": 1 / (first + 1) if first is not None else 0.0,
        "ndcg@10": dcg / idcg if idcg else 0.0,
    }


def evaluate(index: SearchIndex, gold: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, cfg in configs.items():
        per_q, lat = [], []
        for g in gold:
            res, ms = timed_search(g["question"], filters=SearchFilters(bid_id=g["bid_id"]), top_k=K,
                                   use_cache=False, index=index, **cfg)
            lat.append(ms)
            ranked = [(e.file_name, e.page_number) for e in res]
            relevant = {(r["file"], r["page"]) for r in g["relevant"]}
            per_q.append({"id": g["id"], "split": g["split"], "kind": g["kind"], **score_ranking(ranked, relevant)})
        out[name] = {"per_question": per_q, "p50_ms": statistics.median(lat)}
    return out


def summarize(results: dict[str, Any], split: str | None = None, kind: str | None = None) -> dict[str, dict]:
    table = {}
    for name, r in results.items():
        rows = [q for q in r["per_question"] if (split is None or q["split"] == split)
                and (kind is None or q["kind"] == kind)]
        metrics = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")
        table[name] = {m: round(statistics.mean(q[m] for q in rows), 3) for m in metrics} | {
            "p50_ms": round(r["p50_ms"]), "n": len(rows)}
    return table


def md_table(table: dict[str, dict], title: str) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "p50_ms", "n"]
    lines = [f"### {title}", "", "| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |" for name, row in table.items()]
    return "\n".join(lines) + "\n"


def chunk_size_study(sizes: list[int], gold: list[dict[str, Any]]) -> dict[str, dict]:
    """Re-chunk already-ingested documents at each size into a separate index; default config only."""
    from ingestion.pipeline import ingest_folder

    docs = [d for bid in sorted({g["bid_id"] for g in gold}) for d in ingest_folder(ROOT / "data" / bid)]
    chunking = settings()["chunking"]
    original = dict(chunking)
    table = {}
    try:
        for size in sizes:
            chunking.update(target_chars=size, overlap_chars=int(size * 0.15))
            index = SearchIndex(store_dir=ROOT / ".index" / f"eval_chunks_{size}", collection=f"eval_{size}")
            for doc in docs:
                index.index_document(doc)
            res = evaluate(index, gold, {"hybrid+expand+rerank": CONFIGS["hybrid+expand+rerank"]})
            row = summarize(res)["hybrid+expand+rerank"]
            table[f"{size} chars (~{size // 4} tok), {index.count()} chunks"] = row
    finally:
        chunking.update(original)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes", default="", help="comma-separated target_chars values to compare")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="subset of configs to run")
    args = ap.parse_args()
    gold = load_gold()
    index = get_index()
    for bid in sorted({g["bid_id"] for g in gold}):
        index.index_folder(ROOT / "data" / bid)
    results = evaluate(index, gold, {k: CONFIGS[k] for k in args.configs.split(",")})

    report = ["# Retrieval Evaluation", "",
              f"Gold set: {len(gold)} questions ({sum(g['split'] == 'tune' for g in gold)} tune / "
              f"{sum(g['split'] == 'heldout' for g in gold)} held-out), page-level relevance, "
              "search filtered by bid_id, k=10.", "",
              "recall@k = share of questions with a relevant page in the top k. "
              "Latency p50 is per query on CPU, cache off.", ""]
    report.append(md_table(summarize(results), "All questions"))
    report.append(md_table(summarize(results, split="tune"), "Tune split"))
    report.append(md_table(summarize(results, split="heldout"), "Held-out split (never used for tuning)"))
    report.append(md_table(summarize(results, kind="exact"), "Exact-value questions (bid/part/model numbers)"))
    if args.chunk_sizes:
        sizes = [int(s) for s in args.chunk_sizes.split(",")]
        report.append(md_table(chunk_size_study(sizes, gold), "Chunk size study (hybrid+expand+rerank)"))
    report.append("## Questions\n\n| id | bid | split | kind | question | relevant (file p.page) |\n"
                  "|---|---|---|---|---|---|")
    report += [f"| {g['id']} | {g['bid_id']} | {g['split']} | {g['kind']} | {g['question']} | "
               + "; ".join(f"{r['file']} p.{r['page']}" for r in g["relevant"]) + " |" for g in gold]
    (ROOT / "eval" / "report.md").write_text("\n".join(report) + "\n")
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=1))
    print("\n".join(report[:7]))
    print(md_table(summarize(results), "All questions"))


if __name__ == "__main__":
    main()
