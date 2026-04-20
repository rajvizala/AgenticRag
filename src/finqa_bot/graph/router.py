"""Question router.

Two backends:
- ``vllm``: reuses the vLLM OpenAI-compatible endpoint with a tiny JSON schema
  for the classification. Cheapest on Colab because we're already paying the
  TTFT for the generator.
- ``transformers``: loads ``Qwen/Qwen3-1.7B`` directly and runs a short
  classification prompt on device. Kept for production deployments where the
  router model is served separately from the generator.

Both backends share a rule-based fallback that inspects surface keywords so
the graph degrades gracefully when no LLM is reachable (offline tests, eval
failures, etc.).
"""

from __future__ import annotations

import re
from typing import Any

from finqa_bot.config import GpuConfig, Settings
from finqa_bot.graph.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from finqa_bot.logging import get_logger
from finqa_bot.serving.openai_client import build_structured_client
from finqa_bot.types import RoutingDecision

log = get_logger(__name__)


RATIO_KEYWORDS = (
    "ratio",
    "percent",
    "percentage",
    "%",
    "proportion",
    "share of",
    "fraction",
    "growth",
    "increase",
    "decrease",
    "change",
)

MULTI_STEP_KEYWORDS = (
    "average",
    "total",
    "sum",
    "between",
    "over the",
    "from ",
    "and ",
    "combined",
    "net",
    "after",
    "before",
)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def rule_route(question: str) -> RoutingDecision:
    """Cheap keyword-based fallback for when no LLM is available."""
    q = question.lower()
    year_count = len(YEAR_RE.findall(question))
    has_ratio = any(k in q for k in RATIO_KEYWORDS)
    has_multi = any(k in q for k in MULTI_STEP_KEYWORDS)

    if "not about" in q or "unrelated" in q or not q.strip():
        return RoutingDecision(category="out_of_scope", route="refuse", confidence=0.6, reason="heuristic")
    if has_ratio and year_count <= 2 and not has_multi:
        return RoutingDecision(category="single_ratio", route="specialist", confidence=0.55, reason="heuristic")
    if has_multi or year_count >= 3:
        return RoutingDecision(category="multi_step", route="generalist", confidence=0.55, reason="heuristic")
    if "what is" in q or "what was" in q:
        return RoutingDecision(category="direct_lookup", route="specialist", confidence=0.5, reason="heuristic")
    return RoutingDecision(category="multi_step", route="generalist", confidence=0.4, reason="default")


class Router:
    """Async classifier returning a :class:`RoutingDecision`."""

    def __init__(self, cfg: GpuConfig, settings: Settings) -> None:
        self.cfg = cfg
        self.settings = settings
        self._tx_pipeline: Any = None
        self._structured_client: Any = None

    async def classify(self, question: str) -> RoutingDecision:
        if not self.cfg.router.enabled:
            return RoutingDecision(
                category="multi_step",
                route="generalist",
                confidence=1.0,
                reason="router_disabled",
            )

        decision: RoutingDecision | None = None

        if self.cfg.router.backend == "vllm":
            client = await self._get_structured_client()
            if client is not None:
                try:
                    messages = [
                        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": ROUTER_USER_TEMPLATE.format(question=question)},
                    ]
                    decision = await client.ainvoke(messages)
                except Exception as exc:
                    log.warning("Router LLM call failed (%s); falling back to rules.", exc)
        elif self.cfg.router.backend == "transformers":
            pipeline = await self._get_transformers_pipeline()
            if pipeline is not None:
                try:
                    decision = _classify_with_transformers(pipeline, question)
                except Exception as exc:
                    log.warning("Router transformers backend failed (%s); falling back to rules.", exc)

        if decision is None:
            decision = rule_route(question)

        # Apply specialist.enabled guard to every routing outcome — both LLM
        # and rule-based paths — so the pipeline never wastes time on 404
        # retries when the specialist vLLM process is not running.
        if decision.route == "specialist" and not self.cfg.specialist.enabled:
            decision = RoutingDecision(
                category=decision.category,
                route="generalist",
                confidence=decision.confidence,
                reason=decision.reason + "+specialist_disabled",
            )
        return decision

    async def _get_structured_client(self) -> Any:
        if self._structured_client is None:
            try:
                # For the vllm backend, route through the served generator alias
                # (e.g. "generator"). llm_router_model names the HuggingFace
                # checkpoint used only by the transformers backend; it is not a
                # valid vLLM served-model-name.
                if self.cfg.router.backend == "vllm":
                    model_name = self.settings.llm_model
                else:
                    model_name = self.settings.llm_router_model or self.settings.llm_model
                self._structured_client = build_structured_client(
                    settings=self.settings,
                    schema=RoutingDecision,
                    model=model_name,
                    temperature=0.0,
                    max_tokens=self.cfg.router.max_new_tokens,
                )
            except Exception as exc:
                log.warning("Could not build structured router client: %s", exc)
                self._structured_client = None
        return self._structured_client

    async def _get_transformers_pipeline(self) -> Any:
        if self._tx_pipeline is not None:
            return self._tx_pipeline
        try:
            import torch
            from transformers import pipeline

            device = self.cfg.router.device
            if device == "cpu":
                # Use device= (not device_map=) with explicit float32 so that
                # transformers does not probe or warm up the CUDA allocator at
                # all. With device_map="cpu" + torch_dtype="auto", some versions
                # of transformers still run caching_allocator_warmup on the
                # visible CUDA device, consuming scarce VRAM.
                pipeline_kwargs: dict = {"device": "cpu", "torch_dtype": torch.float32}
            else:
                pipeline_kwargs = {"device_map": device, "torch_dtype": "auto"}

            self._tx_pipeline = pipeline(
                "text-generation",
                model=self.cfg.router.model,
                trust_remote_code=True,
                **pipeline_kwargs,
            )
        except Exception as exc:
            log.warning("Failed to load transformers router (%s); disabling.", exc)
            self._tx_pipeline = None
        return self._tx_pipeline


def _classify_with_transformers(pipeline: Any, question: str) -> RoutingDecision:
    prompt = (
        ROUTER_SYSTEM_PROMPT
        + "\n"
        + ROUTER_USER_TEMPLATE.format(question=question)
    )
    out = pipeline(prompt, max_new_tokens=64, do_sample=False, return_full_text=False)
    text = (out[0].get("generated_text") or "").strip()
    import json

    try:
        data = json.loads(text)
        return RoutingDecision(**data)
    except Exception:
        return rule_route(question)
