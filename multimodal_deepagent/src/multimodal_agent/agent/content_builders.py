"""HumanMessage content block builders for different image delivery modes."""

from __future__ import annotations


def build_content_base64(
    text: str,
    image_b64s: list[str] | None = None,
    video_frame_b64s: list[str] | None = None,
    image_urls: list[str] | None = None,
) -> list[dict] | str:
    """Build HumanMessage content using base64 data URLs for the vision model.

    SeaweedFS/web URLs are appended to the text block so MCP tools can use them.
    """
    if not image_b64s and not video_frame_b64s and not image_urls:
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

    blocks: list[dict] = [{"type": "text", "text": url_text}]
    for i, b64 in enumerate(image_b64s or []):
        blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for i, url in enumerate(image_urls or []):
        blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    if video_frame_b64s:
        blocks.append({"type": "text", "text": "[Video frames]"})
        for b64 in video_frame_b64s:
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return blocks


def build_content_seaweedfs(
    text: str,
    image_urls: list[str] | None = None,
    video_frame_urls: list[str] | None = None,
) -> list[dict] | str:
    """Build HumanMessage content using SeaweedFS pre-signed URLs.

    Both the LLM vision blocks and the MCP tool text use the same URLs.
    Requires the LLM endpoint to be able to fetch the SeaweedFS URLs.
    """
    if not image_urls and not video_frame_urls:
        return text

    url_text = text + "\n\n"
    if image_urls:
        url_text += f"[{len(image_urls)} image(s) uploaded — pre-signed URLs:]\n"
        for i, url in enumerate(image_urls):
            url_text += f"  Image {i + 1}: {url}\n"
        url_text += "Pass these URLs to any MCP tool that requires image input.\n\n"
    if video_frame_urls:
        url_text += f"[Video — {len(video_frame_urls)} frame(s) uploaded:]\n"
        for i, url in enumerate(video_frame_urls):
            url_text += f"  Frame {i}: {url}\n"
        url_text += "Pass these URLs to any MCP tool that requires image input.\n"

    blocks: list[dict] = [{"type": "text", "text": url_text}]
    for i, url in enumerate(image_urls or []):
        blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    if video_frame_urls:
        blocks.append({"type": "text", "text": "[Video frames]"})
        for url in video_frame_urls:
            blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks
