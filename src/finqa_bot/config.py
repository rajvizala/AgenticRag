"""Configuration loading.

Two layers:
1. ``Settings`` (pydantic-settings) - runtime env vars: URLs, paths, feature
   toggles, secrets. Loaded from ``.env`` or the process environment.
2. ``GpuConfig`` and ``EvalConfig`` - structured YAML files that live in
   ``configs/``. They describe the GPU profile (T4 vs L4) and the evaluation
   slices + ablations. Typed with Pydantic so mistakes surface at load time.

The GPU config path is chosen automatically by ``setup.sh`` based on
``nvidia-smi`` output and passed via ``FINQA_GPU_CONFIG``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-level settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    vllm_base_url: str = Field(default="http://127.0.0.1:8000/v1")
    vllm_api_key: str = Field(default="EMPTY")

    llm_model: str = Field(default="generator")
    llm_specialist_model: str = Field(default="rLLM/rLLM-FinQA-4B")
    llm_router_model: str = Field(default="Qwen/Qwen3-1.7B")

    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    reranker_model: str = Field(default="Qwen/Qwen3-Reranker-0.6B")
    retrieval_k: int = Field(default=8)
    rerank_k: int = Field(default=4)

    finqa_log_level: str = Field(default="INFO")
    finqa_data_dir: Path = Field(default=Path("./data"))
    finqa_runs_dir: Path = Field(default=Path("./runs"))
    finqa_checkpoint_db: Path = Field(default=Path("./data/checkpoints/graph.sqlite"))
    finqa_gpu_config: Path = Field(default=Path("configs/t4.yaml"))
    finqa_eval_config: Path = Field(default=Path("configs/eval.yaml"))
    finqa_api_host: str = Field(default="0.0.0.0")
    finqa_api_port: int = Field(default=8001)
    finqa_gradio_port: int = Field(default=7860)

    openai_api_key: str = Field(default="")
    openai_baseline_model: str = Field(default="gpt-4o-mini")

    hf_home: Path | None = Field(default=None)
    huggingface_hub_token: str = Field(default="")

    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="finqa-bot")
    langchain_api_key: str = Field(default="")


# ----- GPU config -------------------------------------------------------------


class GpuSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arch: Literal["turing", "ampere", "ada", "hopper", "blackwell"]
    vram_gb: int


class VllmGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    quantization: str | None = None
    # GGUF-specific fields. When gguf_file is set the launcher emits
    # --load-format gguf --gguf-file <gguf_file> and optionally
    # --tokenizer <tokenizer> (required when the GGUF repo ships no tokenizer).
    gguf_file: str | None = None
    tokenizer: str | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 4096
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enforce_eager: bool = False
    served_model_name: str = "generator"
    extra_args: list[str] = Field(default_factory=list)


class VllmSpecialistConfig(VllmGeneratorConfig):
    enabled: bool = False


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = "Qwen/Qwen3-1.7B"
    backend: Literal["transformers", "vllm"] = "transformers"
    device: str = "cuda"
    max_new_tokens: int = 16


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    device: str = "cuda"
    batch_size: int = 32


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    device: str = "cuda"
    batch_size: int = 8


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieval_k: int = 8
    rerank_k: int = 4
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    rrf_k: int = 60


class SelfConsistencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    samples: int = 3
    sampling_temperature: float = 0.6


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 0.0
    top_p: float = 1.0
    self_consistency: SelfConsistencyConfig = Field(default_factory=SelfConsistencyConfig)


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 8000
    api_host: str = "127.0.0.1"
    api_port: int = 8001
    gradio_port: int = 7860
    tunnel: Literal["cloudflared", "gradio_share", "none"] = "cloudflared"


class GpuConfig(BaseModel):
    """Complete GPU-aware config, mirrors ``configs/t4.yaml`` / ``configs/l4.yaml``."""

    model_config = ConfigDict(extra="forbid")

    gpu: GpuSpec
    generator: VllmGeneratorConfig
    specialist: VllmSpecialistConfig
    router: RouterConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    serving: ServingConfig


# ----- Eval config -----------------------------------------------------------


class SliceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: Literal["train", "dev", "test"]
    n: int | None = None
    seed: int = 13
    description: str = ""
    filter: str | None = None
    perturbation: str | None = None


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_accuracy: bool = True
    program_accuracy: bool = True
    retrieval_recall_at_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    full_retrieval_recall_at_k: list[int] = Field(default_factory=lambda: [3, 5, 10])
    unit_correctness: bool = True
    groundedness_rate: bool = True
    refusal_rate: bool = True
    latency_percentiles: list[int] = Field(default_factory=lambda: [50, 90, 95, 99])
    cost_per_correct: bool = True


class AblationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)


class BatchingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eval_concurrency: int = 8
    max_retries: int = 2
    timeout_s: int = 120


class BootstrapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence_level: float = 0.95
    resamples: int = 1000
    seed: int = 13


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    splits: dict[str, str]
    slices: dict[str, SliceSpec]
    metrics: MetricsConfig
    ablations: list[AblationSpec]
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)


# ----- Loaders ---------------------------------------------------------------


_SETTINGS_CACHE: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or refresh:
        _SETTINGS_CACHE = Settings()
    return _SETTINGS_CACHE


def load_gpu_config(path: Path | str | None = None) -> GpuConfig:
    """Load and validate a GPU YAML config.

    Defaults to the path held in ``Settings.finqa_gpu_config`` if ``path`` is
    ``None``.
    """
    settings = Settings()
    resolved = Path(path) if path is not None else settings.finqa_gpu_config
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"GPU config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return GpuConfig(**raw)


def load_eval_config(path: Path | str = Path("configs/eval.yaml")) -> EvalConfig:
    """Load and validate the eval config."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Eval config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return EvalConfig(**raw)


def apply_overrides(cfg: GpuConfig, overrides: dict[str, Any]) -> GpuConfig:
    """Apply dotted-path overrides to a ``GpuConfig`` and return a new instance.

    Keys look like ``generator.enable_prefix_caching`` or
    ``generation.self_consistency.samples``. The ``pipeline.*`` namespace is
    reserved for boolean feature toggles consumed by the graph nodes; those are
    stored in the returned ``GpuConfig.extra`` via ``model_copy(update=...)``
    escape hatch on a separate pipeline object.
    """
    data = cfg.model_dump()
    for dotted, value in overrides.items():
        if dotted.startswith("pipeline."):
            # Pipeline toggles are handled outside the GpuConfig by the graph
            # builder; skip them here so the config validator still passes.
            continue
        if dotted.startswith("llm."):
            # Baseline-OpenAI style overrides surface via Settings / env; skip.
            continue
        _set_dotted(data, dotted, value)
    return GpuConfig(**data)


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            raise KeyError(f"Cannot override {dotted}: path segment '{part}' is not a dict.")
        cursor = nxt
    cursor[parts[-1]] = value


def pipeline_flags(overrides: dict[str, Any]) -> dict[str, bool]:
    """Extract the ``pipeline.*`` toggles from an overrides dict.

    These toggles are interpreted by the LangGraph builder, not the GpuConfig.
    """
    out: dict[str, bool] = {}
    for dotted, value in overrides.items():
        if not dotted.startswith("pipeline."):
            continue
        key = dotted.removeprefix("pipeline.")
        out[key] = bool(value)
    return out
