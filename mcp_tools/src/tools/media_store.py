"""
Unified media resolver for the MCP server.

Provides a ``resolve_media_source`` function that accepts either:
- A URL to an image/video (http:// or https://)
- A base64-encoded image/video string (with or without data URL prefix)

For images it returns a list containing a single PIL Image.
For videos it extracts evenly-spaced frames and returns them as a list of PIL Images.

This allows all downstream tools (detection, classification) to accept media
directly without requiring a separate upload step.
"""

import base64
import logging
import os
import re
import tempfile
from io import BytesIO
from typing import Literal

import cv2
import requests
from PIL import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-media-store")

# Maximum number of frames to extract from a video
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "8"))

# Timeout for downloading media from URLs
DOWNLOAD_TIMEOUT = int(os.environ.get("MEDIA_DOWNLOAD_TIMEOUT", "60"))


def _is_url(source: str) -> bool:
    """Check if the source string is a URL."""
    return source.startswith("http://") or source.startswith("https://")


def _is_base64(source: str) -> bool:
    """Check if the source string looks like base64 data."""
    # Check for data URL prefix
    if source.startswith("data:"):
        return True
    # Check if it's a valid base64 string (rough check)
    # Base64 strings are alphanumeric with +, /, and = padding
    if len(source) > 100:  # Base64 images are typically long
        try:
            # Try to decode a small portion to validate
            base64.b64decode(source[:100] + "==", validate=True)
            return True
        except Exception:
            pass
    return False


def _extract_base64_data(source: str) -> tuple[str, str]:
    """
    Extract the raw base64 data and media type from a source string.
    
    Returns (base64_data, media_type) where media_type is 'image' or 'video'.
    """
    # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
    if source.startswith("data:"):
        match = re.match(r"data:([^;]+);base64,(.+)", source, re.DOTALL)
        if match:
            mime_type = match.group(1)
            b64_data = match.group(2)
            media_type = "video" if mime_type.startswith("video/") else "image"
            return b64_data, media_type
    
    # Raw base64 - assume image by default
    return source, "image"


def _download_media(url: str) -> tuple[bytes, str]:
    """
    Download media from a URL.
    
    Returns (bytes, media_type) where media_type is 'image' or 'video'.
    """
    logger.info("Downloading media from URL: %s", url[:100])
    
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
    response.raise_for_status()
    
    content_type = response.headers.get("Content-Type", "").lower()
    
    # Determine media type from content-type header
    if "video" in content_type:
        media_type = "video"
    elif "image" in content_type:
        media_type = "image"
    else:
        # Try to guess from URL extension
        url_lower = url.lower()
        if any(ext in url_lower for ext in [".mp4", ".avi", ".mov", ".webm", ".mkv"]):
            media_type = "video"
        else:
            media_type = "image"
    
    data = response.content
    logger.info("Downloaded %d bytes, detected media type: %s", len(data), media_type)
    
    return data, media_type


def _bytes_to_pil_image(data: bytes) -> PILImage.Image:
    """Convert raw bytes to a PIL Image."""
    return PILImage.open(BytesIO(data)).convert("RGB")


def _extract_video_frames(
    video_bytes: bytes,
    n_frames: int = 8,
) -> list[PILImage.Image]:
    """
    Extract evenly-spaced frames from raw video bytes.

    Writes the bytes to a temporary file so OpenCV can read them, then
    samples ``n_frames`` frames and returns them as PIL Images.
    """
    # Write to a temp file because cv2.VideoCapture needs a file path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            logger.warning("Video has 0 frames — returning empty list")
            return []

        n_frames = min(n_frames, total)
        indices = [int(i * total / n_frames) for i in range(n_frames)]
        frames: list[PILImage.Image] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # OpenCV returns BGR; convert to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = PILImage.fromarray(rgb_frame)
            frames.append(pil_frame)

        cap.release()
        return frames

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def resolve_media_source(
    source: str,
    max_frames: int | None = None,
) -> tuple[Literal["image", "video"], list[PILImage.Image]]:
    """
    Resolve media from a URL or base64 string.

    Parameters
    ----------
    source : str
        Either:
        - A URL (http:// or https://) pointing to an image or video
        - A base64-encoded string (with or without data URL prefix)
    max_frames : int | None
        For videos, the maximum number of frames to extract.
        Defaults to ``MAX_VIDEO_FRAMES`` env var (8).

    Returns
    -------
    tuple[Literal["image", "video"], list[PILImage.Image]]
        - The media type (``"image"`` or ``"video"``).
        - A list of PIL Images.  For images this is a single-element list.
          For videos this contains up to ``max_frames`` evenly-spaced frames.

    Raises
    ------
    ValueError
        If the source is neither a valid URL nor valid base64 data.
    """
    if max_frames is None:
        max_frames = MAX_VIDEO_FRAMES

    # Handle URL
    if _is_url(source):
        try:
            data, media_type = _download_media(source)
        except requests.RequestException as e:
            raise ValueError(f"Failed to download media from URL: {e}") from e
        
        if media_type == "video":
            frames = _extract_video_frames(data, max_frames)
            logger.info("Resolved URL as VIDEO (%d frame(s) extracted)", len(frames))
            return "video", frames
        else:
            pil_image = _bytes_to_pil_image(data)
            logger.info(
                "Resolved URL as IMAGE (%dx%d)",
                pil_image.width, pil_image.height,
            )
            return "image", [pil_image]

    # Handle base64
    if _is_base64(source):
        b64_data, media_type = _extract_base64_data(source)
        
        try:
            data = base64.b64decode(b64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {e}") from e
        
        if media_type == "video":
            frames = _extract_video_frames(data, max_frames)
            logger.info("Resolved base64 as VIDEO (%d frame(s) extracted)", len(frames))
            return "video", frames
        else:
            pil_image = _bytes_to_pil_image(data)
            logger.info(
                "Resolved base64 as IMAGE (%dx%d)",
                pil_image.width, pil_image.height,
            )
            return "image", [pil_image]

    raise ValueError(
        "Invalid media source. Must be either a URL (http:// or https://) "
        "or a base64-encoded string (with or without data URL prefix)."
    )
