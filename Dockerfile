# A4000専用のllama.cppを作成した。

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG LLAMA_CPP_RELEASE=llama-cpp-cuda-12.8-a4000
ARG LLAMA_CPP_TARBALL_URL=https://github.com/K2iwai/runpod-model/releases/download/${LLAMA_CPP_RELEASE}/${LLAMA_CPP_RELEASE}.tar.gz

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir "huggingface_hub[hf_transfer]"

WORKDIR /app

RUN curl -fsSL -o /tmp/llama-cpp.tar.gz "${LLAMA_CPP_TARBALL_URL}" \
    && mkdir -p /app/bin \
    && tar xzf /tmp/llama-cpp.tar.gz -C /app/bin --strip-components=1 \
    && rm /tmp/llama-cpp.tar.gz \
    && chmod +x /app/bin/llama-server

ENV LD_LIBRARY_PATH=/app/bin
ENV PATH=/app/bin:${PATH}
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

CMD ["/app/entrypoint.sh"]
