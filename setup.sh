#!/usr/bin/env bash
# One-command bootstrap for the FinQA chatbot.
#
# Usage (Colab / Linux):
#   bash setup.sh                 # install, start vLLM + API + UI
#   bash setup.sh --ingest-only   # download FinQA + build retrieval index, then exit
#   bash setup.sh --no-vllm       # skip vLLM (smoke testing / CPU-only path)
#   bash setup.sh --share         # tunnel the Gradio app publicly
#   bash setup.sh --l4            # force L4 config even if GPU is misidentified
#   bash setup.sh --t4            # force T4 config
#   bash setup.sh --fin-o1        # use configs/l4_fino1.yaml (Fin-o1-8B on L4)
#   bash setup.sh --persist-gdrive
#   bash setup.sh --persist-dir=/content/drive/MyDrive/finqa_bot_cache
#
# What this script does, in order:
#   1. Pick a GPU config (T4 vs L4) by reading ``nvidia-smi --query-gpu=name``.
#   2. pip-install the project (runtime + serve extras) idempotently.
#   3. Download the FinQA dataset and build the hybrid retrieval index.
#   4. Boot ``vllm serve`` in the background with the right flags for the GPU.
#   5. Boot the FastAPI SSE server and the Gradio UI.
#   6. Print the URL to the UI, plus tailing instructions for the logs.
#
# Each long-lived step writes its stdout/stderr to ``data/logs/<step>.log`` so
# reviewers can tail the startup sequence without scrollback loss.

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/data/logs"
PID_DIR="${REPO_ROOT}/data/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

INGEST_ONLY=0
SKIP_VLLM=0
SHARE=0
FORCE_TIER=""
FORCE_CONFIG=""
PERSIST_DIR="${FINQA_PERSIST_DIR:-}"

for arg in "$@"; do
    case "$arg" in
        --ingest-only) INGEST_ONLY=1 ;;
        --no-vllm)     SKIP_VLLM=1 ;;
        --share)       SHARE=1 ;;
        --l4)          FORCE_TIER="l4" ;;
        --t4)          FORCE_TIER="t4" ;;
        --fin-o1)      FORCE_TIER="l4"; FORCE_CONFIG="configs/l4_fino1.yaml" ;;
        --persist-gdrive) PERSIST_DIR="/content/drive/MyDrive/finqa_bot_cache" ;;
        --persist-dir=*) PERSIST_DIR="${arg#--persist-dir=}" ;;
        --help|-h)
            sed -n '1,25p' "$0"
            exit 0
            ;;
        *) echo "[setup] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log() { printf "[setup %s] %s\n" "$(date -u +'%H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

_copy_tree() {
    local src="$1"
    local dst="$2"
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$src/" "$dst/"
    else
        cp -a "$src/." "$dst/"
    fi
}

_restore_persisted_data() {
    if [[ -z "$PERSIST_DIR" ]]; then
        return
    fi
    local p="$PERSIST_DIR"
    local local_data="$REPO_ROOT/data"
    local persisted_index="$p/indices/dev"
    local local_index="$local_data/indices/dev"
    local persisted_raw="$p/raw/finqa/dev.json"
    local local_raw="$local_data/raw/finqa/dev.json"

    log "Persistence enabled: $p"
    mkdir -p "$p"

    if [[ -d "$persisted_index" ]]; then
        log "Restoring cached index from $persisted_index"
        mkdir -p "$local_index"
        _copy_tree "$persisted_index" "$local_index"
    else
        log "No persisted index found at $persisted_index"
    fi

    if [[ -f "$persisted_raw" ]]; then
        log "Restoring cached dev split from $persisted_raw"
        mkdir -p "$(dirname "$local_raw")"
        cp -f "$persisted_raw" "$local_raw"
    fi
}

_sync_persisted_data() {
    if [[ -z "$PERSIST_DIR" ]]; then
        return
    fi
    local p="$PERSIST_DIR"
    local local_data="$REPO_ROOT/data"
    local persisted_index="$p/indices/dev"
    local local_index="$local_data/indices/dev"
    local persisted_raw="$p/raw/finqa/dev.json"
    local local_raw="$local_data/raw/finqa/dev.json"

    mkdir -p "$p/indices/dev" "$p/raw/finqa"

    if [[ -d "$local_index" ]]; then
        log "Syncing built index to $persisted_index"
        _copy_tree "$local_index" "$persisted_index"
    fi

    if [[ -f "$local_raw" ]]; then
        cp -f "$local_raw" "$persisted_raw"
        log "Synced dev split to $persisted_raw"
    fi
}

# --- Colab detection ----------------------------------------------------------
_detect_colab() {
    # Returns 0 (true) if running inside Google Colab.
    if [[ -n "${COLAB_RELEASE_TAG:-}" ]] || [[ -d "/content" && -f "/usr/local/lib/python3.10/dist-packages/google/colab/__init__.py" ]] || [[ -d "/content" && -f "/usr/local/lib/python3.11/dist-packages/google/colab/__init__.py" ]]; then
        return 0
    fi
    return 1
}

IS_COLAB=0
if _detect_colab; then
    IS_COLAB=1
    log "Detected Google Colab environment"
    # Auto-enable share in Colab so users get a public URL
    if [[ "$SHARE" -eq 0 ]]; then
        SHARE=1
        log "Auto-enabling --share for Colab (localhost is not accessible)"
    fi
fi

_check_port_free() {
    local port="$1"
    if command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | grep -q ":${port} "; then
        return 1
    fi
    return 0
}

_wait_for_api_ready() {
    local port="$1"
    local timeout="${2:-240}"
    local waited=0
    log "Waiting for FastAPI to be ready on port ${port} (up to ${timeout}s) ..."
    while [[ "$waited" -lt "$timeout" ]]; do
        HEALTH_JSON=$(curl -fs "http://127.0.0.1:${port}/health" 2>/dev/null || echo "")
        if echo "$HEALTH_JSON" | grep -q '"runner":[[:space:]]*true'; then
            log "FastAPI is ready (after ${waited}s)."
            return 0
        fi
        if [[ "$waited" -gt 0 && $((waited % 30)) -eq 0 ]]; then
            LAST_API_LINE=$(tail -n1 "$LOG_DIR/api.log" 2>/dev/null || echo "(no output yet)")
            log "Still waiting for FastAPI (${waited}s elapsed)... last API log: ${LAST_API_LINE}"
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "WARNING: FastAPI did not become ready within ${timeout}s."
    if [[ -s "$LOG_DIR/api.log" ]]; then
        log "Last 20 lines of api.log:"
        tail -n20 "$LOG_DIR/api.log" 2>/dev/null || true
    else
        log "api.log is empty."
    fi
    return 1
}

_detect_gpu() {
    if [[ -n "${FORCE_TIER}" ]]; then
        echo "$FORCE_TIER"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "none"
        return
    fi
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
    case "$name" in
        *L4*)    echo "l4" ;;
        *A100*)  echo "l4" ;;
        *H100*)  echo "l4" ;;
        *T4*)    echo "t4" ;;
        *)       echo "${name:-none}" ;;
    esac
}

_pick_config() {
    local tier="$1"
    if [[ -n "${FORCE_CONFIG}" ]]; then
        if [[ ! -f "${FORCE_CONFIG}" ]]; then
            die "Forced config not found: ${FORCE_CONFIG}"
        fi
        echo "${FORCE_CONFIG}"
        return
    fi
    case "$tier" in
        l4)   echo "configs/l4.yaml" ;;
        t4)   echo "configs/t4.yaml" ;;
        none) die "No NVIDIA GPU detected. Set FORCE_TIER or use --no-vllm to run on CPU-only (reduced scope)." ;;
        *)    log "Unknown GPU '$tier'; defaulting to T4 profile."; echo "configs/t4.yaml" ;;
    esac
}

TIER=$(_detect_gpu)
CONFIG_PATH=$(_pick_config "$TIER")
log "GPU tier: ${TIER} -> ${CONFIG_PATH}"

export FINQA_GPU_CONFIG="${CONFIG_PATH}"
if [[ ! -f ".env" && -f ".env.example" ]]; then
    cp .env.example .env
    sed -i "s|^FINQA_GPU_CONFIG=.*|FINQA_GPU_CONFIG=${CONFIG_PATH}|" .env || true
fi

PY="${PY:-python}"
log "Using Python: $(command -v "$PY" || echo "$PY")"

log "Installing project (runtime)"
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet -e .

if [[ "$SKIP_VLLM" -eq 0 && "$TIER" != "none" ]]; then
    log "Installing vLLM (serve extras)"
    $PY -m pip install --quiet -e ".[serve]" || log "vLLM install failed; continuing without vllm."
fi

_restore_persisted_data

log "Downloading FinQA dataset + building hybrid index"
$PY -m finqa_bot.cli ingest --split dev | tee "$LOG_DIR/ingest.log"

_sync_persisted_data

if [[ "$INGEST_ONLY" -eq 1 ]]; then
    log "Ingest complete. Exiting per --ingest-only."
    exit 0
fi

_start_bg() {
    # Optional first argument: --detach
    #   Wraps the command with setsid, placing it in a new Linux session.
    #   Processes in a new session are NOT in the parent's process group, so
    #   SIGINT/SIGTERM sent to the parent shell (e.g. Colab Stop button) never
    #   reach them.  Use --detach for services that must survive cell interrupts
    #   (namely vLLM, which holds the GPU and is expensive to restart).
    local use_setsid=0
    if [[ "${1:-}" == "--detach" ]]; then
        use_setsid=1
        shift
    fi
    local name="$1"; shift
    local cmd=("$@")
    local log_path="$LOG_DIR/${name}.log"
    local pid_path="$PID_DIR/${name}.pid"
    log "Starting ${name}: ${cmd[*]}"
    if [[ "$use_setsid" -eq 1 ]]; then
        # exec inside the subshell avoids an extra fork; setsid creates a new
        # session before exec-ing into nohup -> cmd.  $! captures the PID of
        # the subshell which, after all the exec chains, is the PID of cmd.
        ( PYTHONUNBUFFERED=1 exec setsid nohup "${cmd[@]}" >"$log_path" 2>&1 ) &
        echo $! >"$pid_path"
    else
        (
            PYTHONUNBUFFERED=1 nohup "${cmd[@]}" >"$log_path" 2>&1 &
            echo $! >"$pid_path"
        )
    fi
    sleep 1
    if [[ -f "$pid_path" ]]; then
        local pid
        pid=$(cat "$pid_path")
        if ! kill -0 "$pid" 2>/dev/null; then
            log "WARNING: ${name} failed to start; see ${log_path}"
        else
            log "${name} PID=${pid} (log: ${log_path})"
        fi
    fi
}

_stop_bg() {
    local name="$1"
    local pid_path="$PID_DIR/${name}.pid"
    if [[ -f "$pid_path" ]]; then
        local pid
        pid=$(cat "$pid_path") || return 0
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping ${name} (PID=${pid})"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_path"
    fi
}

trap 'log "Caught interrupt. Setup script exiting. vLLM / API / UI continue running in the background. Run eval or other commands in a new cell."; exit 0' INT TERM

# --- Clean up any previous run -----------------------------------------------
# On re-runs (common in Colab), old processes hold GPU memory and ports.
# Kill everything first so the new run starts fresh.
_cleanup_previous() {
    local had_old=0
    # Kill by PID file first (clean shutdown)
    for svc in vllm api ui; do
        local pf="$PID_DIR/${svc}.pid"
        if [[ -f "$pf" ]]; then
            local old_pid
            old_pid=$(cat "$pf" 2>/dev/null) || continue
            if kill -0 "$old_pid" 2>/dev/null; then
                log "Killing old ${svc} (PID=${old_pid}) from previous run"
                kill "$old_pid" 2>/dev/null || true
                had_old=1
            fi
            rm -f "$pf"
        fi
    done
    # Also kill by process pattern (catches orphans)
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "finqa_bot.cli api" 2>/dev/null || true
    pkill -f "finqa_bot.cli demo" 2>/dev/null || true

    if [[ "$had_old" -eq 1 ]]; then
        log "Waiting for old processes to release GPU memory ..."
        sleep 5
        # Force-kill anything still hanging
        pkill -9 -f "vllm" 2>/dev/null || true
        pkill -9 -f "finqa_bot.cli api" 2>/dev/null || true
        pkill -9 -f "finqa_bot.cli demo" 2>/dev/null || true
        sleep 3
    fi
}

_cleanup_previous

if [[ "$SKIP_VLLM" -eq 0 && "$TIER" != "none" ]]; then
    VLLM_PORT=8000
    if ! _check_port_free "$VLLM_PORT"; then
        log "Port ${VLLM_PORT} still in use after cleanup; waiting ..."
        sleep 5
    fi
    $PY -m finqa_bot.cli doctor >"$LOG_DIR/doctor.log" 2>&1 || true
    # Build the launcher script with GPU-appropriate flags.
    $PY - <<'PY'
from finqa_bot.config import Settings, load_gpu_config
from finqa_bot.serving.vllm_launcher import write_launcher_script
cfg = load_gpu_config()
out = write_launcher_script(cfg)
print(out)
PY

    # Sync LLM_MODEL in .env to the served model alias so the API client uses
    # the right name when calling vLLM. vLLM registers the model under
    # --served-model-name (e.g. "generator"), not the HuggingFace model ID.
    SERVED_MODEL_NAME=$($PY -c "from finqa_bot.config import load_gpu_config; print(load_gpu_config().generator.served_model_name)" 2>/dev/null || echo "generator")
    export LLM_MODEL="${SERVED_MODEL_NAME}"
    if [[ -f ".env" ]]; then
        if grep -q "^LLM_MODEL=" .env; then
            sed -i "s|^LLM_MODEL=.*|LLM_MODEL=${SERVED_MODEL_NAME}|" .env
        else
            echo "LLM_MODEL=${SERVED_MODEL_NAME}" >> .env
        fi
        log "Set LLM_MODEL=${SERVED_MODEL_NAME} in .env (matches vLLM --served-model-name)"
    fi

    _start_bg --detach vllm bash "$REPO_ROOT/scripts/run_vllm.sh"

    # Wait for vLLM to be fully ready.
    # On first run this includes model download (~2-5 min) + model loading (~2-4 min).
    # Total timeout: 600s = 10 minutes.
    # Initial 10s delay: avoid false-positive from a dying old process.
    log "Waiting for vLLM to be ready (this may take 5-10 min on first run) ..."
    sleep 10
    VLLM_READY=0
    VLLM_WAIT_SECS=10
    VLLM_TIMEOUT=600
    while [[ "$VLLM_WAIT_SECS" -lt "$VLLM_TIMEOUT" ]]; do
        # Show progress every 30 seconds by tailing the last line of vllm.log
        if [[ $((VLLM_WAIT_SECS % 30)) -eq 0 && "$VLLM_WAIT_SECS" -gt 0 ]]; then
            LAST_LINE=$(tail -n1 "$LOG_DIR/vllm.log" 2>/dev/null || echo "(no output yet)")
            log "Still waiting for vLLM (${VLLM_WAIT_SECS}s elapsed)... last log: ${LAST_LINE}"
        fi
        # Check /v1/models for actual model availability (not just /health)
        if curl -fs http://127.0.0.1:${VLLM_PORT}/v1/models 2>/dev/null | grep -q '"id"'; then
            log "vLLM is healthy and model is loaded (after ${VLLM_WAIT_SECS}s)."
            VLLM_READY=1
            break
        fi
        sleep 5
        VLLM_WAIT_SECS=$((VLLM_WAIT_SECS + 5))
    done
    if [[ "$VLLM_READY" -eq 0 ]]; then
        log "WARNING: vLLM did not fully start within ${VLLM_TIMEOUT}s."
        log "Last 5 lines of vllm.log:"
        tail -n5 "$LOG_DIR/vllm.log" 2>/dev/null || true
        log "The API and UI will start anyway; queries will retry until vLLM is ready."
    fi

    # Extra warm-up: send a trivial completion to force full model initialization
    if [[ "$VLLM_READY" -eq 1 ]]; then
        log "Warming up vLLM with a test request ..."
        WARMUP_MODEL=$($PY -c "from finqa_bot.config import load_gpu_config; print(load_gpu_config().generator.served_model_name)" 2>/dev/null || echo "generator")
        WARMUP_STATUS=$(curl -s -o "$LOG_DIR/vllm_warmup.log" -w "%{http_code}" http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${WARMUP_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            || echo "curl_failed")
        if [[ "$WARMUP_STATUS" == "200" ]]; then
            log "vLLM warm-up complete."
        else
            log "WARNING: vLLM warm-up returned status ${WARMUP_STATUS}."
            log "Warm-up response body:"
            tail -n20 "$LOG_DIR/vllm_warmup.log" 2>/dev/null || true
        fi
    fi
fi

API_PORT="${FINQA_API_PORT:-8001}"
GRADIO_PORT="${FINQA_GRADIO_PORT:-7860}"

# On T4 (16 GB) vLLM's 8B model consumes almost all VRAM.  Any CUDA context
# opened by a second process — even one whose models are on CPU — reserves
# ~1.5 GiB of device memory that the EngineCore then cannot use for activation
# buffers, causing OOM on the first real forward pass.  Hiding the GPU
# prevents that parasitic allocation entirely.
#
# On L4/A100 (24+ GB) there is enough VRAM for vLLM (14B AWQ ≈ 9 GB weights)
# plus the auxiliary API-side models (embedder + reranker + router ≈ 2.5 GB).
# Hiding the GPU would break those models, so we keep CUDA visible there.
if [[ "$TIER" == "t4" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    log "T4: Set CUDA_VISIBLE_DEVICES='' for API/UI processes (vLLM owns the GPU)"
else
    log "L4+: keeping CUDA visible for API/UI (embedder/reranker/router run on GPU)"
fi

if ! _check_port_free "$API_PORT"; then
    log "Port ${API_PORT} already in use; skipping API startup. Set FINQA_API_PORT to override."
else
    _start_bg api $PY -m finqa_bot.cli api --host 0.0.0.0 --port "$API_PORT"
    _wait_for_api_ready "$API_PORT" 240 || true
fi

SHARE_FLAG=""
if [[ "$SHARE" -eq 1 ]]; then SHARE_FLAG="--share"; fi

if ! _check_port_free "$GRADIO_PORT"; then
    log "Port ${GRADIO_PORT} already in use; skipping Gradio startup."
else
    _start_bg ui $PY -m finqa_bot.cli demo --port "$GRADIO_PORT" $SHARE_FLAG
fi

# --- Wait for the Gradio share URL -------------------------------------------
# Gradio needs time to start up and create the share tunnel.
# We poll the UI log for up to 120 seconds looking for the *.gradio.live URL.
GRADIO_URL="http://127.0.0.1:${GRADIO_PORT}/"
if [[ "$SHARE" -eq 1 ]]; then
    log "Waiting for Gradio share URL (up to 120s) ..."
    for i in $(seq 1 24); do
        SHARE_URL=$(grep -oP 'https://[a-z0-9-]+\.gradio\.live' "$LOG_DIR/ui.log" 2>/dev/null | head -n1 || echo "")
        if [[ -n "$SHARE_URL" ]]; then
            GRADIO_URL="${SHARE_URL}"
            log "Got Gradio share URL: ${SHARE_URL}"
            break
        fi
        # Show progress every 30 seconds
        if [[ $((i % 6)) -eq 0 ]]; then
            LAST_UI_LINE=$(tail -n1 "$LOG_DIR/ui.log" 2>/dev/null || echo "(no output yet)")
            UI_PID=$(cat "$PID_DIR/ui.pid" 2>/dev/null || echo "")
            UI_ALIVE="no"
            if [[ -n "$UI_PID" ]] && kill -0 "$UI_PID" 2>/dev/null; then
                UI_ALIVE="yes"
            fi
            LOCAL_UI="down"
            if curl -fsI "http://127.0.0.1:${GRADIO_PORT}/" >/dev/null 2>&1; then
                LOCAL_UI="up"
            fi
            UI_ERROR=$(grep -m1 -E 'Traceback|Error:|Exception:|RuntimeError|ValueError|Failed to create share link' "$LOG_DIR/ui.log" 2>/dev/null || echo "")
            ELAPSED=$((i * 5))
            log "Still waiting for share URL (${ELAPSED}s elapsed)... ui_pid_alive=${UI_ALIVE} local_ui=${LOCAL_UI} last UI log: ${LAST_UI_LINE}"
            if [[ -n "$UI_ERROR" ]]; then
                log "UI error hint: ${UI_ERROR}"
            fi
        fi
        sleep 5
    done
    if [[ "$GRADIO_URL" == "http://127.0.0.1:${GRADIO_PORT}/" ]]; then
        log "WARNING: Could not obtain Gradio share URL from logs."
        UI_PID=$(cat "$PID_DIR/ui.pid" 2>/dev/null || echo "")
        if [[ -n "$UI_PID" ]] && kill -0 "$UI_PID" 2>/dev/null; then
            log "UI process is still running (PID=${UI_PID})."
        else
            log "UI process is not running."
        fi
        if curl -fsI "http://127.0.0.1:${GRADIO_PORT}/" >/dev/null 2>&1; then
            log "Local Gradio endpoint is reachable at http://127.0.0.1:${GRADIO_PORT}/ but no public share URL was emitted."
        else
            log "Local Gradio endpoint is not reachable at http://127.0.0.1:${GRADIO_PORT}/."
        fi
        log "Check ui.log manually: tail -f ${LOG_DIR}/ui.log"
        if [[ -s "$LOG_DIR/ui.log" ]]; then
            log "Last 20 lines of ui.log:"
            tail -n20 "$LOG_DIR/ui.log" 2>/dev/null || true
        else
            log "ui.log is empty."
        fi
    fi
fi

cat <<EOF

=====================================================================
 FinQA bot is up.
 GPU tier       : ${TIER}
 Config         : ${CONFIG_PATH}
 Persistence    : ${PERSIST_DIR:-disabled}
 vLLM endpoint  : http://127.0.0.1:8000/v1
 FastAPI        : http://127.0.0.1:${API_PORT}/health
 Gradio UI      : ${GRADIO_URL}
 Logs           : ${LOG_DIR}/

 Tail live:      tail -f ${LOG_DIR}/vllm.log
 Tear down:      pkill -F ${PID_DIR}/vllm.pid; pkill -F ${PID_DIR}/api.pid; pkill -F ${PID_DIR}/ui.pid

 COLAB WORKFLOW
 --------------
 All three services are running in the background via nohup.
 You can now STOP THIS CELL (click the square Stop button).
 vLLM, the FastAPI, and the Gradio UI will keep running.
 Open a new cell and run:
   !python -m finqa_bot.cli eval --split dev --n 200
=====================================================================
EOF

# Keep the cell alive so Colab does not garbage-collect the runtime.
# Stop the cell at any point once you have seen the banner above.
# The background services (vLLM / API / UI) survive the interrupt.
while true; do sleep 60; done
