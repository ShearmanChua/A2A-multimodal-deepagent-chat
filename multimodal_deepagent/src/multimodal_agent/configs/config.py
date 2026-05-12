"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://rag-mcp-server-1:8000")
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "8"))

# "base64"       - pass images as data URLs directly to the LLM
# "object_store" - upload to an S3-compatible object store and use pre-signed URLs
IMAGE_MODE = os.environ.get("IMAGE_MODE", "base64").lower()

MEMORIES_DIR = Path(os.environ.get("MEMORIES_DIR", "/app/src/memories"))
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", "/app/src/skills"))
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "/app/uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


def object_store_available() -> bool:
    """Return True when an S3-compatible object store endpoint is configured."""
    return bool(os.environ.get("OBJECT_STORE_ENDPOINT"))
