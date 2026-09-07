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

download_model() {
    local model_path

    if [[ "$MODEL_REPO" != */* ]]; then
        echo "MODEL_REPO must be in 'org/name' format: $MODEL_REPO" >&2
        return 1
    fi

    mkdir -p "$MODEL_DIR"
    model_path="${MODEL_DIR}/${MODEL_FILE}"

    echo "Downloading model ${MODEL_REPO}/${MODEL_FILE} (revision: ${MODEL_REVISION})"

    hf download "$MODEL_REPO" "$MODEL_FILE" \
        --revision "$MODEL_REVISION" \
        --local-dir "$MODEL_DIR"

    if [[ ! -s "$model_path" ]]; then
        echo "Download failed: ${model_path}" >&2
        return 1
    fi

    echo "Downloaded model to ${model_path}"
    echo "$model_path"
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    model_path="$MODEL_PATH"
else
    model_path="$(download_model)"
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
