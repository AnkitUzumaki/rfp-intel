"""LLM access for all agents: OpenAI-compatible providers (Gemini / Groq / Ollama / OpenAI ...) from env.

Robustness: JSON-only responses validated against a Pydantic schema, one repair turn on bad JSON,
backoff on 429/timeout/5xx, then fallback provider. Never called with raw documents: the evidence
guard caps prompt size (search-as-tool rule, PDF §7).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from settings import cfg

if TYPE_CHECKING:
    from agents.tracing import Span

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)


class LLMError(RuntimeError):
    """All providers/attempts failed; callers mark the affected work as failed instead of crashing."""


class GuardrailError(RuntimeError):
    """Prompt violates the search-as-tool rule (too much raw document text)."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    small_model: str
    api_key: str


def providers_from_env() -> list[Provider]:
    out = []
    if os.getenv("LLM_MODEL"):
        out.append(Provider("primary", os.getenv("LLM_BASE_URL", ""), os.environ["LLM_MODEL"],
                            os.getenv("LLM_SMALL_MODEL") or os.environ["LLM_MODEL"],
                            os.getenv("LLM_API_KEY") or "not-needed"))  # Ollama/vLLM accept any key
    if os.getenv("FALLBACK_MODEL") and os.getenv("FALLBACK_API_KEY"):
        out.append(Provider("fallback", os.getenv("FALLBACK_BASE_URL", ""), os.environ["FALLBACK_MODEL"],
                            os.getenv("FALLBACK_SMALL_MODEL") or os.environ["FALLBACK_MODEL"],
                            os.environ["FALLBACK_API_KEY"]))
    return out


def _cost(model: str, prompt: int, completion: int) -> float:
    prices = cfg("llm", "prices")
    p_in, p_out = prices.get(model, prices["default"])
    return (prompt * p_in + completion * p_out) / 1_000_000


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ``` fences or add prose; keep the outermost object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class LLM:
    def __init__(self, providers: list[Provider] | None = None, client_factory: Any = openai.OpenAI):
        self.providers = providers if providers is not None else providers_from_env()
        self._factory = client_factory
        self._clients: dict[str, Any] = {}

    def _client(self, p: Provider) -> Any:
        if p.name not in self._clients:
            self._clients[p.name] = self._factory(base_url=p.base_url or None, api_key=p.api_key, max_retries=0,
                                                  timeout=cfg("llm", "timeout_s"))
        return self._clients[p.name]

    def json(self, *, system: str, user: str, schema: type[T], span: Span | None = None, small: bool = False,
             purpose: str = "") -> T:
        if len(user) > cfg("agents", "max_prompt_chars") + 8000:
            raise GuardrailError(f"prompt has {len(user)} chars; agents must send retrieved evidence only")
        if not self.providers:
            raise LLMError("no LLM provider configured (set LLM_MODEL / LLM_BASE_URL / LLM_API_KEY in .env)")
        schema_txt = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        system_msg = f"{system}\n\nReturn ONLY one JSON object that validates against this JSON schema:\n{schema_txt}"
        errors: list[str] = []
        for p in self.providers:
            model = p.small_model if small else p.model
            messages: list[dict[str, str]] = [{"role": "system", "content": system_msg},
                                              {"role": "user", "content": user}]
            repaired = False
            for attempt in range(cfg("llm", "max_attempts")):
                start = time.perf_counter()
                call: dict[str, Any] = {"provider": p.name, "model": model, "purpose": purpose, "attempt": attempt + 1}
                try:
                    resp = self._client(p).chat.completions.create(
                        model=model, messages=messages, temperature=cfg("llm", "temperature"),
                        response_format={"type": "json_object"},
                    )
                except _RETRYABLE as exc:
                    self._record(span, call, start, ok=False, error=type(exc).__name__)
                    errors.append(f"{p.name}/{model}: {type(exc).__name__}")
                    time.sleep(cfg("llm", "backoff_s") * 2 ** attempt)
                    continue
                except openai.APIStatusError as exc:  # 4xx other than 429: this provider won't recover
                    self._record(span, call, start, ok=False, error=f"HTTP {exc.status_code}")
                    errors.append(f"{p.name}/{model}: HTTP {exc.status_code}")
                    break
                usage = getattr(resp, "usage", None)
                call["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                call["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
                content = resp.choices[0].message.content or ""
                try:
                    parsed = schema.model_validate_json(_extract_json(content))
                except ValidationError as exc:
                    self._record(span, call, start, ok=False, error="invalid_json")
                    errors.append(f"{p.name}/{model}: invalid JSON")
                    if repaired:
                        break
                    repaired = True
                    messages += [{"role": "assistant", "content": content},
                                 {"role": "user", "content": "Your reply was not valid for the schema: "
                                  f"{str(exc)[:800]}\nReturn ONLY the corrected JSON object."}]
                    continue
                self._record(span, call, start, ok=True)
                return parsed
        raise LLMError("; ".join(errors) or "LLM call failed")

    @staticmethod
    def _record(span: Span | None, call: dict[str, Any], start: float, ok: bool, error: str | None = None) -> None:
        call.update(ok=ok, error=error, latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    cost_usd=_cost(call["model"], call.get("prompt_tokens", 0), call.get("completion_tokens", 0)))
        if not ok:
            log.warning("llm call failed: %s", call)
        if span is not None:
            span.llm(**call)


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM()
