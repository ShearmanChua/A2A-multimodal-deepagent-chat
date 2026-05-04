"""
In-memory video store for the MCP server.

Allows clients to upload base64-encoded video files and receive a short
``video_id`` back.  Tools can then retrieve the video bytes by ID instead
of requiring the LLM to pass enormous base64 strings as tool arguments.
"""

import base64
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-video-store")

# Recognised video MIME-type prefixes / magic bytes for basic validation
_VIDEO_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x00\x00\x00\x18ftypmp4", "mp4"),
    (b"\x00\x00\x00\x1cftypmp4", "mp4"),
    (b"\x00\x00\x00\x20ftypmp4", "mp4"),
    (b"\x00\x00\x00\x18ftypisom", "mp4/isom"),
    (b"\x00\x00\x00\x1cftypisom", "mp4/isom"),
    (b"\x00\x00\x00\x20ftypisom", "mp4/isom"),
    (b"\x1aE\xdf\xa3", "mkv/webm"),
    (b"RIFF", "avi"),
    (b"\x00\x00\x01\xba", "mpeg"),
    (b"\x00\x00\x01\xb3", "mpeg"),
]

# In-memory store: video_id -> raw video bytes
_store: dict[str, bytes] = {}


def _looks_like_video(data: bytes) -> bool:
    """Quick heuristic check that *data* starts with a known video signature."""
    for sig, _ in _VIDEO_SIGNATURES:
        if data[: len(sig)] == sig:
            return True
    # Also accept ftyp box at any of the common offsets
    if b"ftyp" in data[:32]:
        return True
    return False


def store_video(video_base64: str) -> str:
    """
    Decode a base64-encoded video, perform a basic sanity check, store it,
    and return a short ``video_id``.

    Parameters
    ----------
    video_base64 : str
        The video as a base64 string.  A ``data:<mime>;base64,`` prefix is
        stripped automatically if present.

    Returns
    -------
    str
        A unique ``video_id`` that can be used to retrieve the video later.
    """
    # Strip optional data-URL prefix
    if "," in video_base64[:200]:
        video_base64 = video_base64.split(",", 1)[1]

    video_bytes = base64.b64decode(video_base64)

    if len(video_bytes) < 8:
        raise ValueError("Uploaded data is too small to be a valid video")

    # Basic validation — warn but don't reject (some formats may not match)
    if not _looks_like_video(video_bytes):
        logger.warning(
            "Uploaded data (%d bytes) does not match known video signatures — "
            "storing anyway",
            len(video_bytes),
        )

    video_id = str(uuid.uuid4())[:8]
    _store[video_id] = video_bytes

    logger.info(
        "Stored video %s (%d bytes, %.2f MB)",
        video_id,
        len(video_bytes),
        len(video_bytes) / (1024 * 1024),
    )

    return video_id


def get_video(video_id: str) -> bytes | None:
    """
    Retrieve stored video bytes by ``video_id``.

    Returns
    -------
    bytes | None
        The raw video bytes, or ``None`` if the ID is not found.
    """
    return _store.get(video_id)


def delete_video(video_id: str) -> bool:
    """
    Remove a video from the store.

    Returns
    -------
    bool
        ``True`` if the video was found and deleted, ``False`` otherwise.
    """
    if video_id in _store:
        del _store[video_id]
        logger.info("Deleted video %s", video_id)
        return True
    return False


def list_videos() -> list[str]:
    """Return a list of all stored video IDs."""
    return list(_store.keys())
