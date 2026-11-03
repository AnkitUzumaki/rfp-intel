"""Go / No-Go Agent (bonus): rule-based decision over the validated bid record and a configurable company
profile (config/company_profile.yaml). The LLM only explains the decision; it never changes it."""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from agents.llm import LLM, GuardrailError, LLMError
from agents.prompts import prompt
from agents.schemas import BidRecord, GoNoGoExplanation
from agents.tracing import Span
from agents.validator_agent import parse_date
from settings import ROOT, cfg

Status = Literal["pass", "fail", "unknown"]


class RuleResult(BaseModel):
    rule: str
    status: Status
    detail: str
    hard: bool = True


class GoNoGo(BaseModel):
    bid_id: str
    decision: Literal["GO", "NO-GO", "REVIEW"]
    rules: list[RuleResult] = Field(default_factory=list)
    explanation: str = ""


def load_profile(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or ROOT / "config" / "company_profile.yaml").read_text())


def _value(record: BidRecord, field: str) -> Any:
    f = record.fields.get(field)
    return f.value if f else None


def _manufacturer(record: BidRecord, profile: dict[str, Any]) -> RuleResult:
    mfgs = _value(record, "MFG for Registration")
    if not mfgs:
        return RuleResult(rule="manufacturer", status="pass", detail="no manufacturer registration required")
    needed = mfgs if isinstance(mfgs, list) else [mfgs]
    ours = {m.lower() for m in profile["authorized_manufacturers"]}
    missing = [m for m in needed if not any(o in str(m).lower() for o in ours)]
    detail = f"requires {needed}; not authorized for {missing}" if missing else f"authorized for {needed}"
    return RuleResult(rule="manufacturer", status="fail" if missing else "pass", detail=detail)


def _bid_bond(record: BidRecord, profile: dict[str, Any]) -> RuleResult:
    bond = _value(record, "Bid Bond Requirement")
    if not bond or re.match(r"not required|none", str(bond), re.I):
        return RuleResult(rule="bid_bond", status="pass", detail="no bid bond required")
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", str(bond))
    if not pct:
        return RuleResult(rule="bid_bond", status="unknown", detail=f"bond terms unclear: {bond}", hard=False)
    ok = float(pct.group(1)) <= profile["max_bid_bond_percent"]
    return RuleResult(rule="bid_bond", status="pass" if ok else "fail",
                      detail=f"{pct.group(1)}% vs our limit {profile['max_bid_bond_percent']}%")


def _installation(record: BidRecord, profile: dict[str, Any]) -> RuleResult:
    inst = _value(record, "Installation")
    required = isinstance(inst, dict) and inst.get("required") is True
    return RuleResult(rule="installation", status="fail" if required and not profile["can_provide_installation"]
                      else "pass", detail="installation services required" if required else "no installation required")


def _deadline(record: BidRecord, profile: dict[str, Any]) -> RuleResult:
    due = parse_date(str(_value(record, "Due Date") or ""))
    as_of = datetime.strptime(profile["as_of"], "%Y-%m-%d").date() if profile.get("as_of") else date.today()
    if due is None:
        return RuleResult(rule="deadline", status="unknown", detail="due date not found", hard=False)
    days = (due.date() - as_of).days
    return RuleResult(rule="deadline", status="pass" if days >= profile["min_days_to_prepare"] else "fail",
                      detail=f"{days} days from {as_of} to due date {due.date()}")


def _delivery(record: BidRecord, profile: dict[str, Any]) -> RuleResult:
    delivery = str(_value(record, "Delivery Date") or "")
    m = re.search(r"(\d+)\s*(?:calendar\s+|business\s+)?days", delivery, re.I)
    if not m:
        return RuleResult(rule="delivery", status="unknown", detail=f"delivery terms: {delivery or 'not found'}",
                          hard=False)
    window, needed = int(m.group(1)), profile["min_delivery_days_needed"]
    return RuleResult(rule="delivery", status="pass" if window >= needed else "fail", hard=False,
                      detail=f"bid requires delivery within {window} days; we need at least {needed}")


def check_rules(record: BidRecord, profile: dict[str, Any]) -> list[RuleResult]:
    return [rule(record, profile) for rule in (_manufacturer, _bid_bond, _installation, _deadline, _delivery)]


def decide(rules: list[RuleResult]) -> Literal["GO", "NO-GO", "REVIEW"]:
    if any(r.status == "fail" and r.hard for r in rules):
        return "NO-GO"
    if any(r.status != "pass" for r in rules):
        return "REVIEW"
    return "GO"


def evaluate_go_no_go(record: BidRecord, profile: dict[str, Any] | None = None, span: Span | None = None,
                      llm: LLM | None = None) -> GoNoGo:
    profile = profile or load_profile()
    rules = check_rules(record, profile)
    result = GoNoGo(bid_id=record.bid_id, decision=decide(rules), rules=rules)
    explanation = "; ".join(f"{r.rule}: {r.status} ({r.detail})" for r in rules)
    if llm is not None:
        try:
            fields = {k: v.value for k, v in record.fields.items() if v.value is not None}
            explanation = llm.json(system=prompt("go_no_go_system"),
                                   user=prompt("go_no_go_user", decision=result.decision,
                                               rules="\n".join(f"- {r.rule}: {r.status} — {r.detail}" for r in rules),
                                               fields=json.dumps(fields)[:6000], profile=json.dumps(profile)),
                                   schema=GoNoGoExplanation, span=span, small=True, purpose="go_no_go").explanation
        except (LLMError, GuardrailError):
            pass  # the rule summary is already a complete explanation
    result.explanation = explanation
    return result


def save_go_no_go(result: GoNoGo) -> Path:
    path = ROOT / cfg("outputs", "dir") / f"{result.bid_id}_go_no_go.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {result.bid_id} — Go / No-Go: **{result.decision}**", "", result.explanation, "",
             "| Rule | Status | Hard | Detail |", "|---|---|---|---|"]
    lines += [f"| {r.rule} | {r.status} | {'yes' if r.hard else 'no'} | {r.detail} |" for r in result.rules]
    path.write_text("\n".join(lines) + "\n")
    return path
