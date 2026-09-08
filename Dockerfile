# A4000専用のllama.cppを作成した。

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG LLAMA_CPP_RELEASE=llama-cpp-cuda-12.8-a4000
ARG LLAMA_CPP_TARBALL_URL=https://github.com/K2iwai/runpod-model/releases/download/${LLAMA_CPP_RELEASE}/${LLAMA_CPP_RELEASE}.tar.gz

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    openssh-server \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir huggingface_hub \
    && mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && printf '%s\n' \
        'PermitRootLogin yes' \
        'PasswordAuthentication no' \
        'PubkeyAuthentication yes' \
        >> /etc/ssh/sshd_config.d/runpod.conf

WORKDIR /app

RUN curl -fsSL -o /tmp/llama-cpp.tar.gz "${LLAMA_CPP_TARBALL_URL}" \
    && mkdir -p /app/bin \
    && tar xzf /tmp/llama-cpp.tar.gz -C /app/bin --strip-components=1 \
    && rm /tmp/llama-cpp.tar.gz \
    && chmod +x /app/bin/llama-server

ENV LD_LIBRARY_PATH=/app/bin
ENV PATH=/app/bin:${PATH}
ENV HF_XET_HIGH_PERFORMANCE=1
# HF_TOKEN is provided at runtime (e.g. RunPod environment variables).

COPY entrypoint.sh gemma4.jinja /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE 22 8080

CMD ["/app/entrypoint.sh"]
