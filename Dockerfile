FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama.cpp
RUN git clone --depth 1 \
    https://github.com/ggml-org/llama.cpp.git \
    /app/llama.cpp

# Build llama.cpp with CUDA support
RUN cmake -S /app/llama.cpp \
    -B /app/llama.cpp/build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/llama.cpp/build \
    --config Release \
    -j$(nproc)

# RunPod SDK
RUN pip3 install --break-system-packages runpod

COPY worker.py /app/worker.py

CMD ["python3", "/app/worker.py"]
