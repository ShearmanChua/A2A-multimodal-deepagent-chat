"""
object store media uploader using boto3 S3-compatible API.

Uploads bytes/files to object store and returns pre-signed GET URLs that can be
passed back to the agent or embedded in tool responses.

Configuration via environment variables:
    OBJECT_STORE_ENDPOINT         – e.g. "seaweedfs:8333" (no scheme)
    OBJECT_STORE_ACCESS_KEY       – access key (default "")
    OBJECT_STORE_SECRET_KEY       – secret key (default "")
    OBJECT_STORE_SECURE           – "true" for HTTPS (default "false")
    OBJECT_STORE_BUCKET           – bucket name (default "media")
    OBJECT_STORE_PRESIGN_EXPIRY   – pre-signed URL expiry in seconds (default 3600)
    OBJECT_STORE_EXTERNAL_ENDPOINT – external host for pre-signed URLs when the
        internal endpoint differs from what clients can reach
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    endpoint = os.environ.get("OBJECT_STORE_ENDPOINT", "seaweedfs:8333")
    access_key = os.environ.get("OBJECT_STORE_ACCESS_KEY", "")
    secret_key = os.environ.get("OBJECT_STORE_SECRET_KEY", "")
    secure = os.environ.get("OBJECT_STORE_SECURE", "false").lower() in ("1", "true", "yes")

    scheme = "https" if secure else "http"
    endpoint_url = f"{scheme}://{endpoint}"
    logger.info("Connecting to object store at %s …", endpoint_url)

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
    external = os.environ.get("OBJECT_STORE_EXTERNAL_ENDPOINT")
    if not external:
        return _get_s3_client()

    secure = os.environ.get("OBJECT_STORE_SECURE", "false").lower() in ("1", "true", "yes")
    scheme = "https" if secure else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{external}",
        aws_access_key_id=os.environ.get("OBJECT_STORE_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("OBJECT_STORE_SECRET_KEY", ""),
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _default_bucket() -> str:
    return os.environ.get("OBJECT_STORE_BUCKET", "media")


def _presign_expiry() -> int:
    return int(os.environ.get("OBJECT_STORE_PRESIGN_EXPIRY", "3600"))


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
    """Upload raw bytes to object store and return a pre-signed GET URL."""
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
        "Uploaded %d bytes → objstore://%s/%s (expiry=%ds)",
        len(data), bucket, object_key, _presign_expiry(),
    )
    return url


def upload_image_bytes(
    data: bytes,
    content_type: str = "image/jpeg",
    prefix: str = "mcp/images",
    bucket: str | None = None,
) -> str:
    """Upload raw image bytes and return a pre-signed URL."""
    ext_map = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
    }
    ext = ext_map.get(content_type, "jpg")
    ts = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    object_key = f"{prefix}/{ts}_{unique}.{ext}"
    return upload_bytes(data, object_key, content_type=content_type, bucket=bucket)


def _parse_object_store_path(path: str) -> tuple[str, str]:
    """Parse a bucket/key path with any URI scheme (e.g. objstore://, https://) into (bucket, key)."""
    without_scheme = path.split("://", 1)[-1]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def get_presigned_url(path: str, expiry: int | None = None) -> str:
    """Generate a presigned GET URL for an objstore:// path reference.

    Args:
        path: objstore:// path, e.g. ``objstore://media/docling/doc/123.png``
        expiry: URL validity in seconds; defaults to OBJECT_STORE_PRESIGN_EXPIRY env var.
    """
    bucket, key = _parse_object_store_path(path)
    expiry = expiry or _presign_expiry()
    url = _get_presign_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry,
    )
    logger.info("Generated presigned URL for %s (expiry=%ds)", path, expiry)
    return url


def get_image_bytes(path: str) -> tuple[bytes, str]:
    """Download an image from object store and return (raw bytes, content_type).

    Args:
        path: objstore:// path, e.g. ``objstore://media/docling/doc/123.png``

    Returns:
        Tuple of (image bytes, MIME type string e.g. "image/png").
    """
    bucket, key = _parse_object_store_path(path)
    response = _get_s3_client().get_object(Bucket=bucket, Key=key)
    img_bytes = response["Body"].read()
    content_type = response.get("ContentType", "image/png")
    logger.info("Retrieved image from %s (%d bytes)", path, len(img_bytes))
    return img_bytes, content_type
