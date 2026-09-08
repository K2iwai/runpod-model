#!/bin/bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-culturerevolt/gemma-4-12b-heretic-abliterated-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-12b-heretic-Q4_K_M.gguf}"
MODEL_REVISION="${MODEL_REVISION:-main}"
MODEL_DIR="${MODEL_DIR:-/tmp/model}"

LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
# Gemma 4 12B Q4_K_M (~7.4 GB) + A4000 16GB: model fits fully on GPU; KV cache is the limit.
# Default: 96K context with Q8 KV (long-context balance). Override for other profiles:
#   quality:  LLAMA_CTX_SIZE=65536  LLAMA_CACHE_TYPE_K=q8_0 LLAMA_CACHE_TYPE_V=q8_0
#   longer:   LLAMA_CTX_SIZE=131072 LLAMA_CACHE_TYPE_K=q4_0 LLAMA_CACHE_TYPE_V=q4_0
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-98304}"
LLAMA_CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-q8_0}"
LLAMA_CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-q8_0}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"
# Web UI works with small history; Cursor resends a large system/tools block every turn.
LLAMA_MODEL_ALIAS="${LLAMA_MODEL_ALIAS:-Gemma-4-12B-heretic}"
# Qwen3.5 reasoning breaks Cursor multi-turn unless disabled at the API layer.
LLAMA_REASONING="${LLAMA_REASONING:-off}"
LLAMA_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-0}"
LLAMA_REASONING_FORMAT="${LLAMA_REASONING_FORMAT:-none}"
LLAMA_REASONING_PRESERVE="${LLAMA_REASONING_PRESERVE:-false}"

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    echo "HF_TOKEN is set; using authenticated Hugging Face download"
fi

setup_ssh() {
    if [[ -z "${PUBLIC_KEY:-}" ]]; then
        echo "PUBLIC_KEY not set; skipping SSH setup" >&2
        return 0
    fi

    echo "Setting up SSH..." >&2
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys

    if [[ ! -f /etc/ssh/ssh_host_rsa_key ]]; then
        ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ''
    fi
    if [[ ! -f /etc/ssh/ssh_host_ed25519_key ]]; then
        ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
    fi

    service ssh start
    echo "SSH server started" >&2
}

setup_ssh
download_model() {
    local model_path

    if [[ "$MODEL_REPO" != */* ]]; then
        echo "MODEL_REPO must be in 'org/name' format: $MODEL_REPO" >&2
        return 1
    fi

    mkdir -p "$MODEL_DIR"
    model_path="${MODEL_DIR}/${MODEL_FILE}"

    echo "Downloading model ${MODEL_REPO}/${MODEL_FILE} (revision: ${MODEL_REVISION})" >&2

    hf download "$MODEL_REPO" "$MODEL_FILE" \
        --revision "$MODEL_REVISION" \
        --local-dir "$MODEL_DIR"

    if [[ ! -s "$model_path" ]]; then
        echo "Download failed: ${model_path}" >&2
        return 1
    fi

    echo "Downloaded model to ${model_path}" >&2
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    model_path="$MODEL_PATH"
else
    download_model
    model_path="${MODEL_DIR}/${MODEL_FILE}"
fi

if [[ ! -f "$model_path" ]]; then
    echo "Model file not found: $model_path" >&2
    exit 1
fi

echo "Starting llama-server with model: $model_path"

exec llama-server \
    --host "$LLAMA_HOST" \
    --port "$LLAMA_PORT" \
    --model "$model_path" \
    --alias "$LLAMA_MODEL_ALIAS" \
    --ctx-size "$LLAMA_CTX_SIZE" \
    --cache-type-k "$LLAMA_CACHE_TYPE_K" \
    --cache-type-v "$LLAMA_CACHE_TYPE_V" \
    --parallel "$LLAMA_PARALLEL" \
    --n-gpu-layers "$LLAMA_N_GPU_LAYERS" \
    --jinja \
    --reasoning "$LLAMA_REASONING" \
    --reasoning-budget "$LLAMA_REASONING_BUDGET" \
    --reasoning-format "$LLAMA_REASONING_FORMAT" \
    $( [ "$LLAMA_REASONING_PRESERVE" = "true" ] && echo --reasoning-preserve || echo --no-reasoning-preserve )
