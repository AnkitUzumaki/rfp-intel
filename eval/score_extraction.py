"""Extraction accuracy per field against eval/extraction_gold.json + confidence calibration.

Score per field in [0, 1] with a rule per field type (exact / fuzzy / datetime / set F1 / keywords /
subfields / products / null). A field counts as correct when score >= 0.8.
Usage: python -m eval.score_extraction  -> eval/extraction_report.md, eval/calibration.md
"""
from __future__ import annotations

import json
import re
import statistics
from difflib import SequenceMatcher
from typing import Any

from agents.schemas import load_fields
from settings import ROOT, cfg

GOLD = ROOT / "eval" / "extraction_gold.json"
CORRECT = 0.8
_NEG = re.compile(r"^(none|not required|no|n/?a|false)\b", re.I)


def norm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def fuzzy(a: Any, b: Any) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [s for s in re.split(r"[;,\n]| and ", str(x)) if s.strip()]


def is_empty(x: Any) -> bool:
    return x is None or x == [] or x == {} or (isinstance(x, str) and bool(_NEG.match(x.strip())))


def set_f1(pred: Any, gold: list[Any], alts: list[Any]) -> float:
    p = as_list(pred)
    if not p:
        return 0.0
    hit_gold = sum(any(fuzzy(g, x) >= 0.8 for x in p) for g in gold)
    hit_pred = sum(any(fuzzy(g, x) >= 0.8 for g in gold + alts) for x in p)
    recall, precision = hit_gold / len(gold), hit_pred / len(p)
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def keywords(pred: Any, words: list[str]) -> float:
    text = json.dumps(pred).lower() if not isinstance(pred, str) else pred.lower()
    return sum(w.lower() in text for w in words) / len(words)


def score_field(field: str, pred: Any, gold: dict[str, Any]) -> float:
    if gold.get("accept_any"):
        return 1.0
    expected = gold.get("value")
    if "keywords" in gold and not isinstance(gold["keywords"], dict):
        if is_empty(pred):
            return 1.0 if gold.get("accept_null") else 0.0
        return keywords(pred, gold["keywords"])
    if expected is None:  # not in documents: correct iff nothing was invented
        return 1.0 if is_empty(pred) else 0.0
    if is_empty(pred):
        return 0.0
    if isinstance(expected, dict):  # subfields
        parts = []
        for k, v in expected.items():
            pv = pred.get(k) if isinstance(pred, dict) else None
            if isinstance(v, bool):
                parts.append(1.0 if pv is v else 0.0)
            else:
                options = [v] + gold.get("alts_by_key", {}).get(k, [])
                parts.append(max(fuzzy(o, pv) for o in options) if pv is not None else 0.0)
        for k, words in (gold.get("keywords") or {}).items():
            parts.append(keywords(pred.get(k) if isinstance(pred, dict) else pred, words))
        return statistics.mean(parts)
    if field == "Product":
        items = pred if isinstance(pred, list) else []
        got = 0.0
        for g in expected:
            match = next((p for p in items if isinstance(p, dict) and fuzzy(g["name"], p.get("name")) >= 0.6), None)
            got += 0 if match is None else 0.5 + 0.5 * (str(match.get("quantity")) == str(g["quantity"]))
        return got / len(expected)
    if isinstance(expected, list):
        return set_f1(pred, expected, gold.get("alts_items", []))
    options = [expected] + gold.get("alts", [])
    rule = load_fields()[field].scoring
    if rule == "datetime":
        return 1.0 if norm(pred) in {norm(o) for o in options} else 0.0
    if rule == "exact":
        return 1.0 if any(norm(o) == norm(pred) or norm(o) in norm(pred) for o in options) else 0.0
    return max(fuzzy(o, pred) for o in options)


def main() -> None:
    gold_all = json.loads(GOLD.read_text())
    out_dir = ROOT / cfg("outputs", "dir")
    rows, calib = [], []
    for bid, gold in ((k, v) for k, v in gold_all.items() if not k.startswith("_")):
        path = out_dir / f"{bid}.json"
        if not path.exists():
            print(f"skip {bid}: run extraction first ({path} missing)")
            continue
        record = json.loads(path.read_text())["fields"]
        for field, g in gold.items():
            pred = record[field]["value"]
            s = score_field(field, pred, g)
            rows.append({"bid": bid, "field": field, "score": round(s, 2), "correct": s >= CORRECT,
                         "addendum": bool(g.get("addendum")), "pred": pred,
                         "expected": g.get("value", g.get("keywords"))})
            calib.append((record[field]["confidence"], s >= CORRECT))
    if not rows:
        return
    lines = ["# Extraction accuracy", "", f"Score per field in [0,1]; correct = score >= {CORRECT}. "
             "Rules per field type: exact, fuzzy, datetime, set F1, keywords, subfields, products, null.", ""]
    for bid in sorted({r["bid"] for r in rows}):
        rs = [r for r in rows if r["bid"] == bid]
        lines.append(f"- **{bid}**: {sum(r['correct'] for r in rs)}/{len(rs)} correct, "
                     f"mean score {statistics.mean(r['score'] for r in rs):.2f}")
    add = [r for r in rows if r["addendum"]]
    if add:
        lines.append(f"- **Addendum-affected fields**: {sum(r['correct'] for r in add)}/{len(add)} correct")
    lines += ["", "| Bid | Field | Score | Correct | Predicted | Expected |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['bid']} | {r['field']}{' (addendum)' if r['addendum'] else ''} | {r['score']} | "
              f"{'✅' if r['correct'] else '❌'} | {json.dumps(r['pred'])[:120].replace('|', '/')} | "
              f"{json.dumps(r['expected'])[:120].replace('|', '/')} |" for r in rows]
    (ROOT / "eval" / "extraction_report.md").write_text("\n".join(lines) + "\n")

    bins = [(0.8, 1.01), (0.5, 0.8), (0.0, 0.5)]
    cal = ["# Confidence calibration", "", f"{len(calib)} fields across bids (small sample: indicative only).", "",
           "| confidence | fields | correct |", "|---|---|---|"]
    for lo, hi in bins:
        inside = [ok for c, ok in calib if lo <= c < hi]
        pct = f"{100 * sum(inside) / len(inside):.0f}%" if inside else "—"
        cal.append(f"| {lo:.1f}–{min(hi, 1.0):.1f} | {len(inside)} | {pct} |")
    (ROOT / "eval" / "calibration.md").write_text("\n".join(cal) + "\n")
    print("\n".join(lines[:8]))
    print("\n".join(cal))


if __name__ == "__main__":
    main()
