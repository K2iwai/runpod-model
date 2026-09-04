# ============================================================
# Build stage
# ============================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 \
    https://github.com/ggml-org/llama.cpp.git \
    /app/llama.cpp

RUN cmake -S /app/llama.cpp \
    -B /app/llama.cpp/build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF

RUN cmake --build /app/llama.cpp/build \
    --config Release \
    -j$(nproc)


# ============================================================
# Runtime stage
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama-server
COPY --from=builder \
    /app/llama.cpp/build/bin/llama-server \
    /app/llama-server

# RunPod SDK
RUN pip3 install --no-cache-dir --break-system-packages runpod

COPY worker.py /app/worker.py

CMD ["python3", "/app/worker.py"]
