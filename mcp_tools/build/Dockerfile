FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install -U pip && pip install uv

WORKDIR /app

RUN uv pip install --system fastmcp ddgs pillow requests ultralytics torch torchvision minio

COPY ./src ./src

EXPOSE 8000

CMD ["uv", "run", "src/server.py"]