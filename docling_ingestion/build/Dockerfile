FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install -U pip uv

# Install PyTorch with CUDA support before other deps so docling's torch
# backend (RapidOCR) gets GPU acceleration.
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY pyproject.toml .
RUN uv pip install --system .

# Download docling models into the image so the container starts without a
# network dependency.  Models land at /app/models to match DOCLING_MODELS_PATH.
RUN docling-tools models download

COPY src/ ./src/

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
