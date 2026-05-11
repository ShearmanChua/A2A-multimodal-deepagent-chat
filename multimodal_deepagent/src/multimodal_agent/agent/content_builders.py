"""HumanMessage content block builders for different image delivery modes."""

from __future__ import annotations


def build_content_base64(
    text: str,
    image_b64s: list[str] | None = None,
    image_urls: list[str] | None = None,
    video_b64: str | None = None,
    video_urls: list[str] | None = None,
) -> list[dict] | str:
    """Build HumanMessage content using base64 data URLs for the vision model.

    SeaweedFS/web URLs are appended to the text block so MCP tools can use them.
    Videos are passed as ``video_url`` blocks (vLLM native format).
    """
    if not image_b64s and not image_urls and not video_b64 and not video_urls:
        return text

    url_text = text
    if image_urls:
        url_text += (
            f"\n\n[{len(image_urls)} image(s) available — "
            "URLs for MCP tools:]\n"
        )
        for i, url in enumerate(image_urls):
            url_text += f"  Image {i + 1}: {url}\n"
        url_text += "Pass these URLs to any MCP tool that requires image input.\n"
    if video_urls:
        url_text += f"\n\n[{len(video_urls)} video(s) available — URLs for MCP tools:]\n"
        for i, url in enumerate(video_urls):
            url_text += f"  Video {i + 1}: {url}\n"
        url_text += "Pass these URLs to any MCP tool that requires video input.\n"

    blocks: list[dict] = [{"type": "text", "text": url_text}]
    for i, b64 in enumerate(image_b64s or []):
        blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for i, url in enumerate(image_urls or []):
        blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    if video_b64:
        blocks.append({"type": "text", "text": "[Video]"})
        blocks.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        })
    for i, url in enumerate(video_urls or []):
        blocks.append({"type": "text", "text": f"[Video {i + 1}]"})
        blocks.append({"type": "video_url", "video_url": {"url": url}})
    return blocks


def build_content_object_store(
    text: str,
    image_urls: list[str] | None = None,
    video_urls: list[str] | None = None,
) -> list[dict] | str:
    """Build HumanMessage content using SeaweedFS pre-signed URLs.

    Both the LLM vision blocks and the MCP tool text use the same URLs.
    Requires the LLM endpoint to be able to fetch the SeaweedFS URLs.
    Videos are passed as ``video_url`` blocks (vLLM native format).
    """
    if not image_urls and not video_urls:
        return text

    url_text = text + "\n\n"
    if image_urls:
        url_text += f"[{len(image_urls)} image(s) uploaded — pre-signed URLs:]\n"
        for i, url in enumerate(image_urls):
            url_text += f"  Image {i + 1}: {url}\n"
        url_text += "Pass these URLs to any MCP tool that requires image input.\n\n"
    if video_urls:
        url_text += f"[{len(video_urls)} video(s) uploaded — URLs for MCP tools:]\n"
        for i, url in enumerate(video_urls):
            url_text += f"  Video {i + 1}: {url}\n"
        url_text += "Pass these URLs to any MCP tool that requires video input.\n"

    blocks: list[dict] = [{"type": "text", "text": url_text}]
    for i, url in enumerate(image_urls or []):
        blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    for i, url in enumerate(video_urls or []):
        blocks.append({"type": "text", "text": f"[Video {i + 1}]"})
        blocks.append({"type": "video_url", "video_url": {"url": url}})
    return blocks
