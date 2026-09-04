#!/bin/bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-mradermacher/Qwen3.5-9B-heretic-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-9B-heretic.Q4_K_M.gguf}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/runpod-volume/huggingface-cache/hub}"

LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"

resolve_hf_cache_model() {
    local org name model_root refs_main snapshots_dir snapshot_hash candidate version

    if [[ "$MODEL_REPO" != */* ]]; then
        echo "MODEL_REPO must be in 'org/name' format: $MODEL_REPO" >&2
        return 1
    fi

    org="${MODEL_REPO%%/*}"
    name="${MODEL_REPO#*/}"
    model_root="${HF_CACHE_ROOT}/models--${org}--${name}"
    refs_main="${model_root}/refs/main"
    snapshots_dir="${model_root}/snapshots"

    if [[ -f "$refs_main" ]]; then
        snapshot_hash="$(tr -d '[:space:]' < "$refs_main")"
        candidate="${snapshots_dir}/${snapshot_hash}"
        if [[ -d "$candidate" ]]; then
            echo "${candidate}/${MODEL_FILE}"
            return 0
        fi
    fi

    if [[ -d "$snapshots_dir" ]]; then
        version="$(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | head -n 1)"
        if [[ -n "$version" ]]; then
            echo "${snapshots_dir}/${version}/${MODEL_FILE}"
            return 0
        fi
    fi

    return 1
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    model_path="$MODEL_PATH"
elif resolved="$(resolve_hf_cache_model)"; then
    model_path="$resolved"
else
    echo "Set MODEL_PATH or mount a Hugging Face cache at ${HF_CACHE_ROOT}." >&2
    exit 1
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
