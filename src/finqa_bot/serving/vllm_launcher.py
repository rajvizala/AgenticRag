"""Build reproducible ``vllm serve`` commands from a :class:`GpuConfig`.

The key surface area here is pinned so the GPU-benchmark docs can be trusted:
- Quantization (``awq``, ``awq_marlin``, ``fp8``, or ``None``).
- GGUF loading via HuggingFace ``repo:quant_type`` format.
- ``--kv-cache-dtype`` (``auto``, ``fp8_e4m3``).
- Prefix caching and chunked prefill toggles.
- ``max_num_seqs`` and ``max_num_batched_tokens`` (continuous batching knobs).
- ``served_model_name`` so LangChain can address the model by a short alias.

We emit both:
- :func:`build_vllm_command` -> ``list[str]`` argv, for programmatic use.
- :func:`write_launcher_script` -> a bash script that ``setup.sh`` sources.

**GGUF handling (vLLM V1 / v0.19+)**

There are two code paths in vLLM for GGUF models:

1. Local path (``/path/to/model.gguf``) — this path calls
   ``maybe_override_with_speculators`` which invokes the ``transformers``
   GGUF parser.  For Qwen3.5, that parser crashes with
   ``ValueError: GGUF model with architecture qwen35 is not supported yet``
   because the ``transformers`` library does not know the ``qwen35``
   architecture identifier.  See: https://github.com/vllm-project/vllm/issues/36456

2. **HuggingFace repo:quant_type format** (``org/repo:Q8_0``) — this path
   bypasses the transformers GGUF parser entirely and works correctly.
   vLLM downloads and caches the specific GGUF file from the HF Hub
   automatically (instant on subsequent restarts).

We always use approach 2 for GGUF models, plus ``--hf-config-path`` and
``--tokenizer`` pointing at the base model HF repo so vLLM can load
``config.json`` and the tokenizer artefacts which the GGUF repo does not ship.

The Turing-specific guardrail (no FP8 or Marlin on T4) is enforced in
:func:`_validate_gpu_compat`. GGUF models bypass the quantization guardrail
because quantisation is embedded inside the GGUF file itself.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from finqa_bot.config import GpuConfig, VllmGeneratorConfig
from finqa_bot.logging import get_logger

log = get_logger(__name__)


VLLM_ENV_EXPORTS = (
    "export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}",
    "export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}",
)


def _vllm_supports_flag(flag: str) -> bool:
    """Check whether the installed vLLM ``serve`` command recognises *flag*.

    vLLM V1 removed several flags (``--disable-log-requests``,
    ``--enable-prefix-caching``, ``--enable-chunked-prefill``) that were
    valid in V0. Rather than hard-coding version checks, we probe the
    ``vllm serve --help`` output for the flag name.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["vllm", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return flag in result.stdout
    except Exception:
        return True


def _validate_gpu_compat(cfg: GpuConfig) -> None:
    """Reject configurations known not to run on the selected GPU arch.

    GGUF models are exempt from the quantization check because quantisation
    is stored inside the GGUF file; no vLLM ``--quantization`` flag is used.
    """
    arch = cfg.gpu.arch
    gen = cfg.generator

    if gen.gguf_file:
        return

    if arch == "turing":
        if gen.quantization in {"awq_marlin", "fp8", "fp8_e4m3", "fp8_e5m2"}:
            raise ValueError(
                f"Quantization '{gen.quantization}' is not supported on Turing (T4). "
                "Use 'awq' on T4 and move the Marlin/FP8 benchmarks to L4."
            )
        if gen.kv_cache_dtype in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            raise ValueError(
                "FP8 KV cache requires Ada+ (L4 or better). Set kv_cache_dtype to 'auto' on T4."
            )


def _gguf_model_arg(model: str, gguf_file: str) -> str:
    """Return the ``repo:quant_type`` model argument for vLLM GGUF loading.

    vLLM V1 requires this format (e.g. ``unsloth/Qwen3.5-9B-GGUF:Q8_0``) to
    bypass the ``transformers`` GGUF architecture parser that crashes on
    ``qwen35`` when a local file path is used instead.

    The quant type is extracted by stripping the model-name prefix and the
    ``.gguf`` suffix from ``gguf_file``:

        model    = "unsloth/Qwen3.5-9B-GGUF"
        gguf_file = "Qwen3.5-9B-Q8_0.gguf"
        → repo_base  = "Qwen3.5-9B"
        → quant_type = "Q8_0"
        → returns    "unsloth/Qwen3.5-9B-GGUF:Q8_0"
    """
    # Strip .gguf suffix
    name = gguf_file[:-5] if gguf_file.endswith(".gguf") else gguf_file

    # Derive base model name from the HF repo (drop -GGUF suffix if present)
    repo_basename = model.rsplit("/", 1)[-1]  # "Qwen3.5-9B-GGUF"
    model_base = (
        repo_basename[:-5] if repo_basename.endswith("-GGUF") else repo_basename
    )  # "Qwen3.5-9B"

    # Strip "ModelBase-" prefix from filename to get the quant type
    prefix = model_base + "-"
    if name.startswith(prefix):
        quant_type = name[len(prefix):]  # "Q8_0"
    else:
        # Fallback: use the last hyphen-separated segment
        quant_type = name.rsplit("-", 1)[-1] if "-" in name else name

    return f"{model}:{quant_type}"


def build_vllm_command(
    cfg: GpuConfig,
    which: str = "generator",
    host: str | None = None,
    port: int | None = None,
) -> list[str]:
    """Return argv for ``vllm serve``.

    ``which`` is ``generator`` or ``specialist``. The specialist is only
    usable if ``cfg.specialist.enabled`` is set.

    For GGUF models the model argument is automatically converted to the
    ``repo:quant_type`` format required by vLLM V1 to avoid the
    ``qwen35 architecture not supported`` crash in the transformers parser.
    """
    _validate_gpu_compat(cfg)
    section: VllmGeneratorConfig
    if which == "generator":
        section = cfg.generator
    elif which == "specialist":
        if not cfg.specialist.enabled:
            raise ValueError("Specialist is disabled in this GPU config.")
        section = cfg.specialist
    else:
        raise ValueError(f"Unknown launcher target: {which}")

    host = host or cfg.serving.host
    port = port if port is not None else cfg.serving.port

    # For GGUF models, convert to repo:quant_type to avoid the transformers
    # GGUF parser crash on unsupported architectures (vLLM issue #36456).
    if section.gguf_file:
        model_arg = _gguf_model_arg(section.model, section.gguf_file)
    else:
        model_arg = section.model

    argv: list[str] = [
        "vllm",
        "serve",
        model_arg,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        section.served_model_name,
        "--max-model-len",
        str(section.max_model_len),
        "--gpu-memory-utilization",
        str(section.gpu_memory_utilization),
        "--max-num-seqs",
        str(section.max_num_seqs),
        "--max-num-batched-tokens",
        str(section.max_num_batched_tokens),
        "--kv-cache-dtype",
        section.kv_cache_dtype,
        "--dtype",
        "auto",
    ]

    if section.gguf_file:
        # The repo:quant_type model arg already tells vLLM to load the GGUF
        # file.  No --load-format flag is needed.
        # --hf-config-path + --tokenizer point to the base HF model which
        # ships config.json and the tokenizer artefacts absent from GGUF repos.
        if section.tokenizer:
            argv += ["--hf-config-path", section.tokenizer]
            argv += ["--tokenizer", section.tokenizer]
    elif section.quantization:
        argv += ["--quantization", section.quantization]

    # vLLM V1 (v0.19+) made prefix-caching / chunked-prefill always-on
    # and removed --disable-log-requests entirely.
    # Only include these flags when the installed version supports them.
    if section.enable_prefix_caching and _vllm_supports_flag("--enable-prefix-caching"):
        argv += ["--enable-prefix-caching"]
    if section.enable_chunked_prefill and _vllm_supports_flag("--enable-chunked-prefill"):
        argv += ["--enable-chunked-prefill"]
    if section.enforce_eager:
        argv += ["--enforce-eager"]
    argv += list(section.extra_args)
    return argv


def write_launcher_script(
    cfg: GpuConfig,
    out_path: Path | str = Path("scripts/run_vllm.sh"),
    log_file: Path | str = Path("data/logs/vllm.log"),
) -> Path:
    """Write a bash script that starts vLLM with the resolved config.

    Idempotent; overwrites any prior file at ``out_path``. Sets executable bit.

    For GGUF models the model argument uses the ``repo:quant_type`` format
    (e.g. ``unsloth/Qwen3.5-9B-GGUF:Q8_0``).  vLLM downloads and caches the
    specific file from HuggingFace Hub automatically; subsequent restarts are
    instant because the file is already in the local cache.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    argv = build_vllm_command(cfg)
    quoted = " \\\n    ".join(_shell_quote(arg) for arg in argv)

    gen = cfg.generator
    display_model = (
        _gguf_model_arg(gen.model, gen.gguf_file) if gen.gguf_file else gen.model
    )

    script = (
        "#!/usr/bin/env bash\n"
        "# Auto-generated by finqa_bot.serving.vllm_launcher; do not edit by hand.\n"
        "set -euo pipefail\n"
        "\n"
    )
    script += "\n".join(VLLM_ENV_EXPORTS) + "\n\n"
    script += (
        f"mkdir -p \"$(dirname '{log_file}')\"\n"
        f"echo \"Starting vLLM: model={display_model}\"\n"
        f"exec {quoted} 2>&1 | tee '{log_file}'\n"
    )
    out_path.write_text(script, encoding="utf-8")
    with contextlib.suppress(OSError):
        out_path.chmod(0o755)
    log.info("Wrote vLLM launcher script to %s", out_path)
    return out_path


def _shell_quote(arg: str) -> str:
    """Minimal POSIX shell quoter; avoids pulling in ``shlex`` at import time."""
    if not arg or any(ch in arg for ch in " \t\n\"'\\$`*?()[]{}|&;<>"):
        escaped = arg.replace("'", "'\"'\"'")
        return f"'{escaped}'"
    return arg


def kill_listeners(port: int) -> None:
    """Best-effort kill of any process bound to ``port``. Linux / macOS only."""
    try:
        if os.name == "nt":
            return
        os.system(f"fuser -k {port}/tcp 2>/dev/null || true")
    except Exception:
        pass
