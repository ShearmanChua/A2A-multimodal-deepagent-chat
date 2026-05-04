"""
In-memory image store for the MCP server.

Allows clients to upload base64-encoded images and receive a short image_id
back. Tools can then retrieve the image bytes by ID instead of requiring the
LLM to pass enormous base64 strings as tool arguments.
"""

import base64
import logging
import uuid
from io import BytesIO

from PIL import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-image-store")

# In-memory store: image_id -> raw image bytes
_store: dict[str, bytes] = {}


def store_image(image_base64: str) -> str:
    """
    Decode a base64-encoded image, validate it, store it, and return a
    short ``image_id``.

    Parameters
    ----------
    image_base64 : str
        The image as a base64 string.  A ``data:<mime>;base64,`` prefix is
        stripped automatically if present.

    Returns
    -------
    str
        A unique ``image_id`` that can be used to retrieve the image later.
    """
    # Strip optional data-URL prefix
    if "," in image_base64[:100]:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    # Validate that the bytes are actually a valid image
    pil_image = PILImage.open(BytesIO(image_bytes))
    pil_image.verify()

    image_id = str(uuid.uuid4())[:8]
    _store[image_id] = image_bytes

    logger.info(
        "Stored image %s (%d bytes, format=%s)",
        image_id,
        len(image_bytes),
        pil_image.format,
    )

    return image_id


def get_image(image_id: str) -> bytes | None:
    """
    Retrieve stored image bytes by ``image_id``.

    Returns
    -------
    bytes | None
        The raw image bytes, or ``None`` if the ID is not found.
    """
    return _store.get(image_id)


def delete_image(image_id: str) -> bool:
    """
    Remove an image from the store.

    Returns
    -------
    bool
        ``True`` if the image was found and deleted, ``False`` otherwise.
    """
    if image_id in _store:
        del _store[image_id]
        logger.info("Deleted image %s", image_id)
        return True
    return False


def list_images() -> list[str]:
    """Return a list of all stored image IDs."""
    return list(_store.keys())
