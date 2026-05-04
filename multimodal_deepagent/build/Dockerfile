FROM python:3.12-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install -U pip && pip install uv

COPY pyproject.toml uv.lock* ./

RUN uv sync --no-install-project

# Copy source code
COPY src ./src

# Install the project itself
RUN uv pip install .

EXPOSE 10010

CMD ["uv", "run", "python", "src/main.py", "--host", "0.0.0.0", "--port", "10010"]
