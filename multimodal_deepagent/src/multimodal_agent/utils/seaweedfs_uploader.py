"""
SeaweedFS media uploader using boto3 S3-compatible API.

Uploads images and video frames to SeaweedFS and returns pre-signed GET URLs
that can be passed to the LLM agent and MCP tools.

Configuration via environment variables:
    SEAWEEDFS_ENDPOINT         – e.g. "seaweedfs:8333" (no scheme)
    SEAWEEDFS_ACCESS_KEY       – access key (default "")
    SEAWEEDFS_SECRET_KEY       – secret key (default "")
    SEAWEEDFS_SECURE           – "true" for HTTPS (default "false")
    SEAWEEDFS_BUCKET           – bucket name (default "media")
    SEAWEEDFS_PRESIGN_EXPIRY   – pre-signed URL expiry in seconds (default 3600)
    SEAWEEDFS_EXTERNAL_ENDPOINT – external host for pre-signed URLs when the
        internal endpoint differs from what clients can reach
        (e.g. "localhost:8333" when running inside Docker)
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    endpoint = os.environ.get("SEAWEEDFS_ENDPOINT", "seaweedfs:8333")
    access_key = os.environ.get("SEAWEEDFS_ACCESS_KEY", "")
    secret_key = os.environ.get("SEAWEEDFS_SECRET_KEY", "")
    secure = os.environ.get("SEAWEEDFS_SECURE", "false").lower() in ("1", "true", "yes")

    scheme = "https" if secure else "http"
    endpoint_url = f"{scheme}://{endpoint}"
    logger.info("Connecting to SeaweedFS S3 at %s …", endpoint_url)

    _s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )
    return _s3_client


def _get_presign_client():
    """Return a client configured for generating pre-signed URLs.

    Uses SEAWEEDFS_EXTERNAL_ENDPOINT when set so that pre-signed URLs are
    reachable from outside the Docker network.
    """
    external = os.environ.get("SEAWEEDFS_EXTERNAL_ENDPOINT")
    if not external:
        return _get_s3_client()

    secure = os.environ.get("SEAWEEDFS_SECURE", "false").lower() in ("1", "true", "yes")
    scheme = "https" if secure else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{external}",
        aws_access_key_id=os.environ.get("SEAWEEDFS_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("SEAWEEDFS_SECRET_KEY", ""),
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _default_bucket() -> str:
    return os.environ.get("SEAWEEDFS_BUCKET", "media")


def _presign_expiry() -> int:
    return int(os.environ.get("SEAWEEDFS_PRESIGN_EXPIRY", "3600"))


def _ensure_bucket(bucket: str) -> None:
    client = _get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        logger.info("Bucket '%s' not found – creating …", bucket)
        client.create_bucket(Bucket=bucket)


def upload_bytes(
    data: bytes,
    object_key: str,
    content_type: str = "application/octet-stream",
    bucket: str | None = None,
) -> str:
    """Upload raw bytes to SeaweedFS and return a pre-signed GET URL."""
    bucket = bucket or _default_bucket()
    _ensure_bucket(bucket)

    client = _get_s3_client()
    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=io.BytesIO(data),
        ContentLength=len(data),
        ContentType=content_type,
    )

    url = _get_presign_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=_presign_expiry(),
    )
    logger.info(
        "Uploaded %d bytes → seaweedfs://%s/%s (expiry=%ds)",
        len(data), bucket, object_key, _presign_expiry(),
    )
    return url


def upload_file(
    file_path: str,
    prefix: str = "uploads",
    bucket: str | None = None,
) -> str:
    """Upload a local file to SeaweedFS and return a pre-signed GET URL."""
    path = Path(file_path)
    ext = path.suffix.lower() or ".bin"
    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or "application/octet-stream"
    ts = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    object_key = f"{prefix}/{ts}_{unique}{ext}"
    return upload_bytes(path.read_bytes(), object_key, content_type=mime, bucket=bucket)


def upload_base64(
    b64_data: str,
    ext: str = "jpg",
    prefix: str = "uploads",
    bucket: str | None = None,
) -> str:
    """Decode a base64 string, upload to SeaweedFS, return a pre-signed URL."""
    if "," in b64_data and b64_data.startswith("data:"):
        header, b64_data = b64_data.split(",", 1)
        if "/" in header:
            mime_part = header.split(";")[0].replace("data:", "")
            guessed_ext = mime_part.split("/")[-1]
            if guessed_ext and guessed_ext != "octet-stream":
                ext = guessed_ext

    data = base64.b64decode(b64_data)

    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "avi": "video/x-msvideo",
        "mkv": "video/x-matroska",
    }
    content_type = mime_map.get(ext, "application/octet-stream")
    ts = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    object_key = f"{prefix}/{ts}_{unique}.{ext}"
    return upload_bytes(data, object_key, content_type=content_type, bucket=bucket)
