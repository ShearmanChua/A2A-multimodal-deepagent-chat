"""
MinIO object-storage tool for the MCP server.

Provides two capabilities:
1. **list_bucket_objects** – list objects (with optional prefix filter) in a
   MinIO bucket.
2. **get_bucket_object** – download an object from a MinIO bucket and return
   it using the correct MCP modality type (``Image`` for images, raw bytes /
   base64 text for everything else).

Configuration is read from environment variables:
    MINIO_ENDPOINT   – e.g. ``minio:9000`` or ``localhost:9000``
    MINIO_ACCESS_KEY – access key (default ``minioadmin``)
    MINIO_SECRET_KEY – secret key (default ``minioadmin``)
    MINIO_SECURE     – ``true`` to use HTTPS (default ``false``)
    MINIO_BUCKET     – default bucket name (default ``data``)
"""

import base64
import logging
import mimetypes
import os
from io import BytesIO
from typing import Any

from minio import Minio
from minio.error import S3Error
from fastmcp.utilities.types import Image  # MCP Image type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-minio-store")

# ---------------------------------------------------------------------------
# MinIO client (lazy singleton)
# ---------------------------------------------------------------------------
_client: Minio | None = None


def _get_client() -> Minio:
    """Return (and lazily create) the MinIO client singleton."""
    global _client
    if _client is not None:
        return _client

    endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    secure = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

    logger.info(
        "Connecting to MinIO at %s (secure=%s) …",
        endpoint,
        secure,
    )

    _client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
    return _client


# ---------------------------------------------------------------------------
# MIME / modality helpers
# ---------------------------------------------------------------------------

# Image MIME types that we can return as MCP Image objects
_IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff")

# Map MIME type to the short format string expected by fastmcp Image
_MIME_TO_FORMAT: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


def _guess_mime(object_name: str) -> str:
    """Guess the MIME type from the object name / extension."""
    mime, _ = mimetypes.guess_type(object_name)
    return mime or "application/octet-stream"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_bucket_objects(
    bucket: str | None = None,
    prefix: str = "",
    recursive: bool = True,
    max_items: int = 100,
) -> list[dict[str, Any]]:
    """
    List objects in a MinIO bucket.

    Parameters
    ----------
    bucket : str | None
        Bucket name.  Falls back to ``MINIO_BUCKET`` env var (default ``data``).
    prefix : str
        Only return objects whose key starts with this prefix.
    recursive : bool
        If ``True`` (default), list recursively through "directories".
    max_items : int
        Maximum number of items to return (default 100).

    Returns
    -------
    list[dict[str, Any]]
        A list of dicts, each containing:
        - ``name`` – full object key
        - ``size`` – size in bytes
        - ``last_modified`` – ISO-8601 timestamp string
        - ``content_type`` – guessed MIME type
    """
    client = _get_client()
    bucket = bucket or os.environ.get("MINIO_BUCKET", "data")

    if not client.bucket_exists(bucket):
        raise ValueError(
            f"Bucket '{bucket}' does not exist. "
            "Create it first via the MinIO console or mc CLI."
        )

    objects: list[dict[str, Any]] = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=recursive):
        if obj.is_dir:
            objects.append({
                "name": obj.object_name,
                "type": "directory",
                "size": 0,
                "last_modified": None,
                "content_type": None,
            })
        else:
            objects.append({
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "content_type": _guess_mime(obj.object_name),
            })

        if len(objects) >= max_items:
            break

    logger.info(
        "Listed %d object(s) in bucket '%s' (prefix='%s').",
        len(objects),
        bucket,
        prefix,
    )
    return objects


def get_bucket_object(
    object_name: str,
    bucket: str | None = None,
) -> Image | dict[str, str]:
    """
    Download an object from a MinIO bucket and return it with the correct
    MCP modality type.

    - **Images** (JPEG, PNG, GIF, WebP, BMP, TIFF) are returned as an MCP
      ``Image`` object so the agent receives them as inline images.
    - **All other files** are returned as a dict with ``filename``,
      ``content_type``, ``size``, and ``data_base64`` (base64-encoded content)
      so the agent can inspect or forward the data.

    Parameters
    ----------
    object_name : str
        The full key / path of the object inside the bucket.
    bucket : str | None
        Bucket name.  Falls back to ``MINIO_BUCKET`` env var (default ``data``).

    Returns
    -------
    Image | dict[str, str]
        An MCP ``Image`` for image files, or a metadata dict with base64 data
        for non-image files.

    Raises
    ------
    ValueError
        If the bucket or object does not exist.
    """
    client = _get_client()
    bucket = bucket or os.environ.get("MINIO_BUCKET", "data")

    try:
        response = client.get_object(bucket, object_name)
        data = response.read()
        response.close()
        response.release_conn()
    except S3Error as exc:
        if exc.code in ("NoSuchBucket", "NoSuchKey"):
            raise ValueError(
                f"Object '{object_name}' not found in bucket '{bucket}'. "
                "Use the list_minio_objects tool to see available objects."
            ) from exc
        raise

    mime = _guess_mime(object_name)
    logger.info(
        "Downloaded '%s' from bucket '%s' (%d bytes, %s).",
        object_name,
        bucket,
        len(data),
        mime,
    )

    # Return as MCP Image if it's an image type
    if mime in _MIME_TO_FORMAT:
        fmt = _MIME_TO_FORMAT[mime]
        return Image(data=data, format=fmt)

    # For non-image files, return metadata + base64 payload
    return {
        "filename": object_name,
        "content_type": mime,
        "size": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
    }
