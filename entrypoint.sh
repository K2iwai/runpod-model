#!/bin/bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-mradermacher/Qwen3.5-9B-heretic-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-9B-heretic.Q4_K_M.gguf}"
MODEL_REVISION="${MODEL_REVISION:-main}"
MODEL_DIR="${MODEL_DIR:-/tmp/model}"

LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"

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
    --ctx-size "$LLAMA_CTX_SIZE" \
    --parallel "$LLAMA_PARALLEL" \
    --n-gpu-layers "$LLAMA_N_GPU_LAYERS"
