"""DeepAgent middleware — pre-model message processing.

Defines ``extract_images_to_human``, which scans ToolMessages for image
content and re-injects it as a HumanMessage with ``image_url`` blocks so
the vision model can see images returned by MCP tools.

Also registers ``parse_messages_before_model`` as a ``@before_model``
middleware when DeepAgents is available.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Matches S3-compatible object store presigned URLs and direct image-extension URLs.
IMAGE_URL_RE = re.compile(
    r'https?://\S+(?:\.(?:jpg|jpeg|png|gif|webp|bmp|tiff)(?:[?#]\S*)?'
    r'|[?&]X-Amz-Signature=[^\s"\'<>]*)',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Core middleware function
# ---------------------------------------------------------------------------


async def extract_images_to_human(messages: list) -> list:
    """Convert MCP ToolMessage image content into a HumanMessage with image_url blocks.

    Handles three formats returned by MCP tools:

    1. **Legacy MCP Image blobs** — ``{"type": "image", "data": ..., "mimeType": ...}``
       in a list-typed ToolMessage content.
    2. **object store URL dicts** — ``{"type": "image_url", "url": ...}`` in a list-typed
       ToolMessage content (returned by ``get_minio_object`` and similar tools).
    3. **String content** — scans for presigned / image-extension URLs with a regex,
       and also parses JSON strings containing an ``image_url`` key.
    """
    new_messages: list = []
    collected_images: list[dict] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            new_messages.append(msg)
            continue

        if isinstance(msg.content, list):
            images_found: list[dict] = []
            remaining: list = []
            for item in msg.content:
                if not isinstance(item, dict):
                    remaining.append(item)
                    continue
                if item.get("type") == "image":
                    # Legacy MCP Image blob — convert to data URL
                    data = item.get("data", "")
                    mime = item.get("mimeType", "image/jpeg")
                    images_found.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    })
                elif item.get("type") == "image_url" and item.get("url"):
                    # Object store presigned URL from MCP tool
                    images_found.append({
                        "type": "image_url",
                        "image_url": {"url": item["url"]},
                    })
                else:
                    remaining.append(item)

            if images_found:
                msg.content = remaining if remaining else "images retrieved"
                collected_images.extend(images_found)

        elif isinstance(msg.content, str):
            # Try JSON first — tool may return a serialised dict
            try:
                parsed = json.loads(msg.content)
                if isinstance(parsed, dict):
                    if parsed.get("type") == "image_url" and parsed.get("url"):
                        collected_images.append({
                            "type": "image_url",
                            "image_url": {"url": parsed["url"]},
                        })
                        msg.content = "image retrieved"
                    elif isinstance(parsed.get("image_url"), str):
                        url = parsed["image_url"]
                        if url.startswith("http"):
                            collected_images.append({
                                "type": "image_url",
                                "image_url": {"url": url},
                            })
                            msg.content = "image retrieved"
            except (json.JSONDecodeError, TypeError):
                # Scan plain-text content for image URLs
                urls = IMAGE_URL_RE.findall(msg.content)
                if urls:
                    for url in urls:
                        collected_images.append({
                            "type": "image_url",
                            "image_url": {"url": url},
                        })
                    msg.content = IMAGE_URL_RE.sub("[image]", msg.content)

        new_messages.append(msg)

    if collected_images:
        new_messages.append(AIMessage(content="(images retrieved)"))
        new_messages.append(HumanMessage(content=collected_images))

    return new_messages


# ---------------------------------------------------------------------------
# DeepAgent @before_model registration
# ---------------------------------------------------------------------------

parse_messages_before_model: Any = None

try:
    from langchain.agents.middleware import AgentState, before_model

    @before_model
    async def parse_messages_before_model(state: AgentState, runtime: Any) -> Any:
        """Middleware that runs before each model call to process tool images."""
        state["messages"] = await extract_images_to_human(state["messages"])
        return state

except ImportError:
    pass
