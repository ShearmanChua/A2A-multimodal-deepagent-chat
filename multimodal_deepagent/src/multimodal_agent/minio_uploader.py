"""
MinIO media uploader using boto3.

Uploads images and video frames to a MinIO bucket and returns pre-signed
URLs that can be passed to the LLM agent and MCP tools.

Configuration via environment variables:
    MINIO_ENDPOINT   – e.g. ``minio:9000`` (without scheme)
    MINIO_ACCESS_KEY – access key (default ``minioadmin``)
    MINIO_SECRET_KEY – secret key (default ``minioadmin``)
    MINIO_SECURE     – ``true`` for HTTPS (default ``false``)
    MINIO_BUCKET     – bucket name (default ``data``)
    MINIO_PRESIGN_EXPIRY – pre-signed URL expiry in seconds (default ``3600``)
    MINIO_EXTERNAL_ENDPOINT – external endpoint for pre-signed URLs if
        different from MINIO_ENDPOINT (e.g. ``localhost:9000`` when running
        inside Docker but accessing from outside)
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

# ---------------------------------------------------------------------------
# Singleton boto3 S3 client
# ---------------------------------------------------------------------------
_s3_client = None


def _get_s3_client():
    """Return (and lazily create) the boto3 S3 client configured for MinIO."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    secure = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

    scheme = "https" if secure else "http"
    endpoint_url = f"{scheme}://{endpoint}"

    logger.info("Connecting to MinIO via boto3 at %s …", endpoint_url)

    _s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",  # MinIO ignores this but boto3 requires it
    )
    return _s3_client


def _get_presign_client():
    """Return a boto3 S3 client configured for generating pre-signed URLs.

    If ``MINIO_EXTERNAL_ENDPOINT`` is set the pre-signed URLs will point to
    that host instead of the internal ``MINIO_ENDPOINT``.  This is useful when
    the agent runs inside Docker but the URLs need to be reachable from the
    host or another network.
    """
    external = os.environ.get("MINIO_EXTERNAL_ENDPOINT")
    if not external:
        return _get_s3_client()

    secure = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
    scheme = "https" if secure else "http"
    endpoint_url = f"{scheme}://{external}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _ensure_bucket(bucket: str) -> None:
    """Create the bucket if it does not already exist."""
    client = _get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError:
        logger.info("Bucket '%s' does not exist – creating …", bucket)
        client.create_bucket(Bucket=bucket)


def _default_bucket() -> str:
    return os.environ.get("MINIO_BUCKET", "data")


def _presign_expiry() -> int:
    return int(os.environ.get("MINIO_PRESIGN_EXPIRY", "3600"))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def upload_bytes(
    data: bytes,
    object_key: str,
    content_type: str = "application/octet-stream",
    bucket: str | None = None,
) -> str:
    """Upload raw bytes to MinIO and return a pre-signed GET URL.

    Parameters
    ----------
    data:
        The raw bytes to upload.
    object_key:
        The S3 object key (path inside the bucket).
    content_type:
        MIME type of the object.
    bucket:
        Bucket name.  Falls back to ``MINIO_BUCKET`` env var.

    Returns
    -------
    str
        A pre-signed URL valid for ``MINIO_PRESIGN_EXPIRY`` seconds.
    """
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

    presign_client = _get_presign_client()
    url = presign_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=_presign_expiry(),
    )

    logger.info(
        "Uploaded %d bytes to s3://%s/%s  → pre-signed URL generated (expiry=%ds)",
        len(data),
        bucket,
        object_key,
        _presign_expiry(),
    )
    return url


def upload_file(
    file_path: str,
    prefix: str = "uploads",
    bucket: str | None = None,
) -> str:
    """Upload a local file to MinIO and return a pre-signed GET URL.

    The object key is ``<prefix>/<timestamp>_<uuid>.<ext>``.
    """
    path = Path(file_path)
    ext = path.suffix.lower() or ".bin"
    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or "application/octet-stream"

    ts = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    object_key = f"{prefix}/{ts}_{unique}{ext}"

    data = path.read_bytes()
    return upload_bytes(data, object_key, content_type=mime, bucket=bucket)


def upload_base64(
    b64_data: str,
    ext: str = "jpg",
    prefix: str = "uploads",
    bucket: str | None = None,
) -> str:
    """Decode a base64 string, upload to MinIO, return a pre-signed URL.

    Automatically strips ``data:<mime>;base64,`` prefixes.
    """
    # Strip data-URL prefix if present
    if "," in b64_data and b64_data.startswith("data:"):
        header, b64_data = b64_data.split(",", 1)
        # Try to extract extension from MIME
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


def upload_video_frames(
    frames_b64: list[str],
    prefix: str = "uploads/frames",
    bucket: str | None = None,
) -> list[str]:
    """Upload a list of base64-encoded JPEG frames and return pre-signed URLs."""
    urls = []
    for i, frame_b64 in enumerate(frames_b64):
        url = upload_base64(
            frame_b64,
            ext="jpg",
            prefix=f"{prefix}/frame_{i:04d}",
            bucket=bucket,
        )
        urls.append(url)
    return urls
