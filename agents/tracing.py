"""Observability: one JSONL record per agent step (input, tool calls, LLM calls, output, tokens, latency)
plus a run-summary footer with totals (tokens, cost, latency). Thread-safe for parallel agents."""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from settings import ROOT, cfg


def _short(value: Any, limit: int = 600) -> Any:
    """Keep traces readable: long strings / lists are truncated, models dumped."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...[+{len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_short(v, limit) for v in value[:20]]
        return items + [f"...[+{len(value) - 20} items]"] if len(value) > 20 else items
    return value


class Span:
    def __init__(self, tracer: Tracer, agent: str, input: Any):
        self.tracer = tracer
        self.record: dict[str, Any] = {
            "type": "step", "run_id": tracer.run_id, "agent": agent, "input": _short(input),
            "tool_calls": [], "llm_calls": [], "output": None, "error": None,
            "started_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }

    def tool(self, name: str, args: Any, result: Any, ms: float) -> None:
        self.record["tool_calls"].append({"tool": name, "args": _short(args, 200), "result": _short(result, 200),
                                          "latency_ms": round(ms, 1)})

    def llm(self, **call: Any) -> None:
        self.record["llm_calls"].append(call)

    def output(self, value: Any) -> None:
        self.record["output"] = _short(value)


class Tracer:
    def __init__(self, mode: str, meta: dict[str, Any] | None = None, out_dir: Path | None = None):
        self.run_id = f"{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.mode, self.meta = mode, meta or {}
        self.path = (out_dir or ROOT / cfg("outputs", "dir") / "traces") / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._start = time.perf_counter()

    @contextmanager
    def span(self, agent: str, input: Any = None) -> Iterator[Span]:
        span = Span(self, agent, input)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.record["error"] = repr(exc)
            raise
        finally:
            rec = span.record
            rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            rec["tokens"] = {
                "prompt": sum(c.get("prompt_tokens", 0) for c in rec["llm_calls"]),
                "completion": sum(c.get("completion_tokens", 0) for c in rec["llm_calls"]),
            }
            rec["cost_usd"] = round(sum(c.get("cost_usd", 0.0) for c in rec["llm_calls"]), 6)
            with self._lock:
                rec["step"] = len(self.records) + 1
                self.records.append(rec)
                with self.path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

    def summary(self) -> dict[str, Any]:
        calls = [c for r in self.records for c in r["llm_calls"]]
        return {
            "type": "run_summary", "run_id": self.run_id, "mode": self.mode, **self.meta,
            "steps": len(self.records),
            "agents": sorted({r["agent"] for r in self.records}),
            "llm_calls": len(calls),
            "llm_failures": sum(not c.get("ok", True) for c in calls),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls),
            "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in calls), 6),
            "tool_calls": sum(len(r["tool_calls"]) for r in self.records),
            "errors": [f"{r['agent']}: {r['error']}" for r in self.records if r["error"]],
            "wall_ms": round((time.perf_counter() - self._start) * 1000, 1),
        }

    def close(self) -> dict[str, Any]:
        summary = self.summary()
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(summary, default=str) + "\n")
        return summary


_TRACERS: dict[str, Tracer] = {}


def new_tracer(mode: str, meta: dict[str, Any] | None = None) -> Tracer:
    tracer = Tracer(mode, meta)
    _TRACERS[tracer.run_id] = tracer
    return tracer


def get_tracer(run_id: str) -> Tracer:
    """Graph state carries only the run_id (serializable); nodes look the tracer up here."""
    return _TRACERS[run_id]
