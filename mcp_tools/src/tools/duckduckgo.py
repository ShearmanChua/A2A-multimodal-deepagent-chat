from ddgs import DDGS
import base64
import requests
from io import BytesIO
from PIL import Image as PILImage
from fastmcp.utilities.types import Image  # ✅ MCP Image type
import json

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

logger = logging.getLogger("mcp-search")


def search_duckduckgo(query, max_results=5):
    results = DDGS().text(query, region='us-en', backend='auto', max_results=max_results)
    logger.info(f"Search google results for {query}: {results}")
    
    return results

def fetch_image_as_base64(url: str, max_size=(512, 512)):
    """
    Download image, compress it, return raw bytes (NOT base64)
    """
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()

        img = PILImage.open(BytesIO(resp.content)).convert("RGB")
        img.thumbnail(max_size)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=70)

        return buffer.getvalue()  # ✅ return bytes instead of base64

    except Exception:
        return None
def search_duckduckgo_images(query: str, max_results: int = 5):
    """
    MCP tool: returns native Image content blocks
    """

    images = []

    logger.info(f"Searching images for: {query}")

    results = DDGS().images(query, region='us-en', backend='auto', max_results=max_results)

    logger.info(f"Search google images for {query}: {results}")

    for r in results:
        img_url = r.get("image")
        logger.info("processing image: %s", img_url)
        if not img_url:
            continue

        img_bytes = fetch_image_as_base64(img_url)
        if not img_bytes:
            continue

        # ✅ MCP-native image object
        images.append(
            Image(
                data=img_bytes,
                format="jpeg"
            )
        )

    return images
