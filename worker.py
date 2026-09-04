"""RunPod Serverless worker: download GGUF at runtime, run llama-server, proxy OpenAI API."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

MODEL_REPO = os.getenv("MODEL_REPO", "mradermacher/Qwen3.5-9B-heretic-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Qwen3.5-9B-heretic.Q4_K_M.gguf")
MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models")

LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = os.getenv("LLAMA_PORT", "8080")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}")
LLAMA_CTX_SIZE = os.getenv("LLAMA_CTX_SIZE", "8192")
LLAMA_PARALLEL = os.getenv("LLAMA_PARALLEL", "1")
LLAMA_N_GPU_LAYERS = os.getenv("LLAMA_N_GPU_LAYERS", "999")
STARTUP_TIMEOUT = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "1800"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))
HEALTH_POLL_INTERVAL = 2

DEFAULT_CHAT_ROUTE = "/v1/chat/completions"
DEFAULT_COMPLETION_ROUTE = "/v1/completions"

llama_process: subprocess.Popen | None = None
_default_model_cache: Optional[str] = None


def ensure_model() -> str:
    """Download a single GGUF file if it is not already on disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.isfile(model_path):
        logging.info("Using cached model: %s", model_path)
        return model_path

    logging.info("Downloading model %s from %s", MODEL_FILE, MODEL_REPO)
    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
    )
    logging.info("Model ready at %s", downloaded)
    return downloaded


def start_llama_server(model_path: str) -> subprocess.Popen:
    argv = [
        "llama-server",
        "--host",
        LLAMA_HOST,
        "--port",
        LLAMA_PORT,
        "--model",
        model_path,
        "--ctx-size",
        LLAMA_CTX_SIZE,
        "--parallel",
        LLAMA_PARALLEL,
        "--n-gpu-layers",
        LLAMA_N_GPU_LAYERS,
    ]
    logging.info("Starting llama-server: %s", " ".join(argv))
    return subprocess.Popen(argv)


def wait_for_llama_server(proc: subprocess.Popen) -> None:
    url = f"{LLAMA_BASE_URL}/health"
    deadline = time.monotonic() + STARTUP_TIMEOUT

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with code {proc.returncode}"
            )
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as resp:
                if resp.status == 200:
                    logging.info("llama-server is healthy")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    raise RuntimeError(f"llama-server did not become healthy within {STARTUP_TIMEOUT}s")


def _forward_signal(signum, _frame):
    logging.info("Received signal %s, shutting down llama-server", signum)
    if llama_process and llama_process.poll() is None:
        llama_process.send_signal(signum)
    sys.exit(128 + signum)


def _is_llama_alive() -> bool:
    return llama_process is None or llama_process.poll() is None


async def _default_model(session: aiohttp.ClientSession) -> Optional[str]:
    global _default_model_cache
    if _default_model_cache is not None:
        return _default_model_cache

    served = os.getenv("SERVED_MODEL_NAME")
    if served:
        _default_model_cache = served.split(",")[0].strip()
        return _default_model_cache

    try:
        async with session.get(f"{LLAMA_BASE_URL}/v1/models") as resp:
            data = await resp.json(content_type=None)
            model_id = (data.get("data") or [{}])[0].get("id")
            if model_id:
                _default_model_cache = model_id
                return model_id
    except Exception as exc:
        logging.warning("Could not resolve default model from /v1/models: %s", exc)

    return MODEL_FILE


def _normalize_job_input(job_input: dict) -> Tuple[str, str, Optional[dict]]:
    if job_input.get("openai_input"):
        return (
            job_input.get("openai_route") or DEFAULT_CHAT_ROUTE,
            "POST",
            job_input["openai_input"],
        )

    if job_input.get("openai_route"):
        return job_input["openai_route"], "GET", None

    if job_input.get("route"):
        body = job_input.get("body")
        method = (job_input.get("method") or ("POST" if body else "GET")).upper()
        return job_input["route"], method, body

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if messages is None and prompt is None:
        raise ValueError(
            "Job input must contain one of: openai_input (+openai_route), "
            "route (+body), or prompt/messages."
        )

    sampling_params = dict(job_input.get("sampling_params") or {})
    body = {**sampling_params, "stream": bool(job_input.get("stream", False))}
    if messages is not None:
        body["messages"] = messages
        return DEFAULT_CHAT_ROUTE, "POST", body
    body["prompt"] = prompt
    return DEFAULT_COMPLETION_ROUTE, "POST", body


def _error(message: str) -> dict:
    return {"error": {"message": message, "type": "worker_error", "code": None}}


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    job_input = job.get("input") or {}

    try:
        route, method, body = _normalize_job_input(job_input)
    except ValueError as exc:
        yield _error(str(exc))
        return

    if not _is_llama_alive():
        yield _error("llama-server process is not running; worker is unhealthy")
        return

    headers = {"Content-Type": "application/json"}

    if method != "GET" and body is not None and "model" not in body:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                model = await _default_model(session)
                if model:
                    body = {**body, "model": model}
        except Exception:
            pass

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.request(
                method, f"{LLAMA_BASE_URL}{route}", json=body
            ) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    logging.error(
                        "llama-server %s %s returned HTTP %s: %s",
                        method,
                        route,
                        resp.status,
                        detail,
                    )
                    yield _error(f"llama-server returned HTTP {resp.status}: {detail}")
                    return

                wants_stream = isinstance(body, dict) and body.get("stream") is True
                if wants_stream:
                    async for chunk in resp.content.iter_any():
                        yield chunk.decode("utf-8", errors="replace")
                else:
                    yield await resp.json(content_type=None)
    except aiohttp.ClientError as exc:
        logging.exception("Request to llama-server failed")
        yield _error(f"Request to llama-server failed: {exc}")


def main() -> None:
    global llama_process

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)

    model_path = ensure_model()
    llama_process = start_llama_server(model_path)
    try:
        wait_for_llama_server(llama_process)
    except RuntimeError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    import runpod

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda _current: 1,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
