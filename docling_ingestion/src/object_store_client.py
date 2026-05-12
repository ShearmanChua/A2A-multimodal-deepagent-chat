"""
object store image uploader for the docling ingestion service.
Mirrors the pattern used in mcp_tools/src/tools/object_store_uploader.py.

Configuration via environment variables:
    OBJECT_STORE_ENDPOINT          – e.g. "seaweedfs:8333"
    OBJECT_STORE_ACCESS_KEY        – access key (default "")
    OBJECT_STORE_SECRET_KEY        – secret key (default "")
    OBJECT_STORE_SECURE            – "true" for HTTPS (default "false")
    OBJECT_STORE_BUCKET            – bucket name (default "media")
    OBJECT_STORE_PRESIGN_EXPIRY    – pre-signed URL expiry in seconds (default 3600)
    OBJECT_STORE_EXTERNAL_ENDPOINT – external host for pre-signed URLs
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger("docling-ingestion.object_store")

_s3_client = None
_presign_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    endpoint = os.environ.get("OBJECT_STORE_ENDPOINT", "seaweedfs:8333")
    access_key = os.environ.get("OBJECT_STORE_ACCESS_KEY", "")
    secret_key = os.environ.get("OBJECT_STORE_SECRET_KEY", "")
    secure = os.environ.get("OBJECT_STORE_SECURE", "false").lower() in ("1", "true", "yes")
    scheme = "https" if secure else "http"
    logger.info("Connecting to object store at %s://%s", scheme, endpoint)
    _s3_client = boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )
    return _s3_client


def _get_presign_client():
    global _presign_client
    if _presign_client is not None:
        return _presign_client
    external = os.environ.get("OBJECT_STORE_EXTERNAL_ENDPOINT")
    if not external:
        return _get_s3_client()
    secure = os.environ.get("OBJECT_STORE_SECURE", "false").lower() in ("1", "true", "yes")
    scheme = "https" if secure else "http"
    _presign_client = boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{external}",
        aws_access_key_id=os.environ.get("OBJECT_STORE_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("OBJECT_STORE_SECRET_KEY", ""),
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )
    return _presign_client


def _ensure_bucket(bucket: str) -> None:
    client = _get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        logger.info("Bucket '%s' not found — creating", bucket)
        client.create_bucket(Bucket=bucket)


def upload_document_image(
    img_bytes: bytes,
    ext: str = "png",
    source_stem: str = "doc",
) -> str:
    """Upload an image extracted from a document and return an objstore:// path reference."""
    bucket = os.environ.get("OBJECT_STORE_BUCKET", "media")

    _ensure_bucket(bucket)

    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    key = f"docling/{source_stem}/{ts}_{uid}.{ext}"
    content_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    _get_s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=io.BytesIO(img_bytes),
        ContentLength=len(img_bytes),
        ContentType=content_type,
    )

    path = f"objstore://{bucket}/{key}"
    logger.info("Uploaded document image → %s (%d bytes)", path, len(img_bytes))
    return path


def get_presigned_url_from_path(path: str) -> str:
    """Generate a presigned GET URL for a path with any URI scheme (e.g. objstore://)."""
    without_scheme = path.split("://", 1)[-1]
    bucket, _, key = without_scheme.partition("/")
    expiry = int(os.environ.get("OBJECT_STORE_PRESIGN_EXPIRY", "3600"))
    return _get_presign_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry,
    )


def get_image_bytes_from_path(path: str) -> bytes:
    """Download image bytes from an objstore:// path reference."""
    without_scheme = path.removeprefix("objstore://")
    bucket, _, key = without_scheme.partition("/")
    response = _get_s3_client().get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def delete_all_objects() -> int:
    """Delete every object in the configured bucket. Returns the count deleted."""
    if not object_store_configured():
        return 0

    bucket = os.environ.get("OBJECT_STORE_BUCKET", "media")
    client = _get_s3_client()
    deleted = 0

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = page.get("Contents", [])
        if not objects:
            continue
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
        )
        deleted += len(objects)
        logger.info("Deleted %d object(s) from bucket '%s'", len(objects), bucket)

    return deleted


def object_store_configured() -> bool:
    """Return True if object store env vars are present enough to try uploading."""
    return bool(os.environ.get("OBJECT_STORE_ENDPOINT"))
