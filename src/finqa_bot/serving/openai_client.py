"""OpenAI-compatible LLM client.

``vLLM`` exposes an OpenAI-compatible server. LangChain's ``ChatOpenAI`` speaks
that protocol out of the box once ``base_url`` is set. This module:

1. Builds :class:`langchain_openai.ChatOpenAI` instances against our vLLM
   endpoint (or optionally against real OpenAI for the baseline ablation).
2. Wires the grammar-constrained structured output via
   ``with_structured_output(schema, method="json_schema", strict=True)``.
3. Exposes a small :func:`wait_for_server` health-check used by ``setup.sh``.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from finqa_bot.config import GpuConfig, Settings
from finqa_bot.logging import get_logger

log = get_logger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)


def build_chat_client(
    settings: Settings,
    model: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 2,
    backend: str = "auto",
) -> Any:
    """Return a configured ``ChatOpenAI`` instance.

    ``backend`` is:
    - ``"auto"`` - use OpenAI if ``settings.openai_api_key`` is set and model
      matches the OpenAI baseline model; otherwise hit the local vLLM server.
    - ``"vllm"`` - force vLLM endpoint.
    - ``"openai"`` - force real OpenAI (requires ``OPENAI_API_KEY``).
    """
    from langchain_openai import ChatOpenAI

    resolved_model = model or settings.llm_model
    if backend == "openai" or (
        backend == "auto"
        and settings.openai_api_key
        and resolved_model == settings.openai_baseline_model
    ):
        log.info("Using real OpenAI endpoint; model=%s", resolved_model)
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_key=settings.openai_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    log.info("Using vLLM endpoint %s; model=%s", settings.vllm_base_url, resolved_model)
    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def build_structured_client(
    settings: Settings,
    schema: type[TSchema],
    model: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 2,
    backend: str = "auto",
) -> Any:
    """Return a ``ChatOpenAI.with_structured_output(schema)`` runnable.

    When pointed at vLLM, the ``json_schema`` response format triggers the
    xgrammar backend under the hood (vLLM v0.12+). The generator is forced to
    produce a JSON document that matches the schema exactly.
    """
    llm = build_chat_client(
        settings=settings,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        backend=backend,
    )
    return llm.with_structured_output(schema, method="json_schema", strict=True)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1.0, max=8.0), reraise=True)
def wait_for_server(
    base_url: str,
    timeout_s: float = 600.0,
    poll_interval_s: float = 2.0,
    api_key: str = "EMPTY",
) -> bool:
    """Block until the vLLM OpenAI-compatible server responds to ``/models``.

    ``base_url`` should already include the ``/v1`` suffix.
    Returns True on success; raises ``TimeoutError`` on timeout.
    """
    deadline = time.time() + timeout_s
    models_url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(models_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("data"):
                        log.info("vLLM healthy at %s (models=%s)", base_url, [m.get("id") for m in data["data"]])
                        return True
        except (httpx.HTTPError, ValueError) as exc:
            last_err = exc
        time.sleep(poll_interval_s)
    msg = f"vLLM did not become ready within {timeout_s}s at {base_url}"
    if last_err is not None:
        msg += f" (last error: {last_err})"
    raise TimeoutError(msg)


def probe_metrics(base_url: str) -> dict[str, float]:
    """Scrape vLLM's Prometheus ``/metrics`` endpoint into a dict.

    Returns a subset of the metrics most useful to the pipeline:
    ``gpu_cache_usage_perc``, ``running_requests``, ``generation_tokens_total``,
    ``time_to_first_token_seconds`` (histogram sum/count), and
    ``prefix_cache_hit_rate``.
    """
    url = base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    interesting = {
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
    }
    out: dict[str, float] = {}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
        for line in resp.text.splitlines():
            if not line or line.startswith("#"):
                continue
            try:
                name, value = line.split(" ", 1)
            except ValueError:
                continue
            base_name = name.split("{", 1)[0]
            if base_name in interesting:
                try:
                    out[base_name] = float(value)
                except ValueError:
                    continue
    except httpx.HTTPError:
        pass
    return out


def healthcheck_probe(settings: Settings) -> dict[str, Any]:
    """Convenience: short JSON-friendly status snapshot."""
    url = settings.vllm_base_url.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(url, headers={"Authorization": f"Bearer {settings.vllm_api_key}"})
            ok = resp.status_code == 200
            data = resp.json() if ok else {}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": ok, "models": [m.get("id") for m in data.get("data", [])]}


def _unused(_cfg: GpuConfig) -> None:
    """Prevent unused-import lint on ``GpuConfig`` while keeping it as a type hint."""
