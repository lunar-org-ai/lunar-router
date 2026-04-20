# ============================================================================
# OpenTracy — Python API server
# ============================================================================
# Usage:
#   docker build -t opentracy-api .
#   docker run -p 8000:8000 opentracy-api
#   docker run --gpus all -p 8000:8000 opentracy-api   # with GPU for training
# ============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git software-properties-common build-essential cmake && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Clone llama.cpp and build the quantize binary for GGUF conversion/quantization
# Includes convert_hf_to_gguf.py script
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF && \
    cmake --build build --target llama-quantize llama-server --config Release -j$(nproc)

WORKDIR /app

# Install PyTorch with CUDA support, then training/inference deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir unsloth && \
    pip install --no-cache-dir gguf sentencepiece protobuf && \
    pip install --no-cache-dir -e ".[server]" 2>/dev/null || \
    pip install --no-cache-dir fastapi uvicorn pydantic clickhouse-connect numpy tqdm httpx openai mcp

COPY opentracy/ opentracy/
COPY clickhouse/ clickhouse/
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e "."

EXPOSE 8000

CMD ["uvicorn", "opentracy.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
