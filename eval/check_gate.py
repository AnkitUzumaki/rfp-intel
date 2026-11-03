"""CI gate: fail the build when retrieval quality drops below a threshold (reads eval/results.json)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from settings import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="recall@5")
    ap.add_argument("--config", default="hybrid+expand+rerank")
    ap.add_argument("--min", type=float, default=0.85)
    args = ap.parse_args()
    results = json.loads((ROOT / "eval" / "results.json").read_text())
    value = statistics.mean(q[args.metric] for q in results[args.config]["per_question"])
    print(f"{args.config} {args.metric} = {value:.3f} (min {args.min})")
    sys.exit(0 if value >= args.min else 1)


if __name__ == "__main__":
    main()
