"""
Test client for the Multimodal DeepAgent A2A server.

Demonstrates:
1. Text-only query
2. Text with system prompt
3. Multiple image upload with query
4. Video upload with query
5. Streaming response

Usage:
    python test_client.py
    python test_client.py --image img1.jpg --image img2.jpg
    python test_client.py --video clip.mp4
    python test_client.py --system "You are a military analyst." --image scene.jpg
    python test_client.py --url http://localhost:10010
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _encode_file_to_b64(file_path: str) -> str:
    """Read a file and return its base64 encoding."""
    return base64.b64encode(Path(file_path).read_bytes()).decode()


def _build_message(
    text: str,
    system_prompt: str | None = None,
    image_paths: list[str] | None = None,
    video_path: str | None = None,
) -> dict[str, Any]:
    """Build an A2A message payload with optional system prompt, images, and video.

    Convention:
    - System prompt is sent as a TextPart prefixed with ``[system]``.
    - Multiple images are sent as separate FileParts.
    - One video is sent as a single FilePart.
    """
    parts: list[dict[str, Any]] = []

    # System prompt (if provided)
    if system_prompt:
        parts.append({
            "kind": "text",
            "text": f"[system] {system_prompt}",
        })

    # User text query
    parts.append({
        "kind": "text",
        "text": text,
    })

    # Images (multiple allowed)
    if image_paths:
        for img_path in image_paths:
            mime, _ = mimetypes.guess_type(img_path)
            mime = mime or "image/jpeg"
            img_b64 = _encode_file_to_b64(img_path)
            parts.append({
                "kind": "file",
                "file": {
                    "bytes": img_b64,
                    "mimeType": mime,
                },
            })

    # Video (one only)
    if video_path:
        mime, _ = mimetypes.guess_type(video_path)
        mime = mime or "video/mp4"
        vid_b64 = _encode_file_to_b64(video_path)
        parts.append({
            "kind": "file",
            "file": {
                "bytes": vid_b64,
                "mimeType": mime,
            },
        })

    return {
        "message": {
            "role": "user",
            "parts": parts,
            "message_id": uuid4().hex,
        },
    }


@click.command()
@click.option("--url", default="http://localhost:10010", help="A2A server URL")
@click.option("--image", "image_paths", multiple=True, help="Path to image file (can specify multiple)")
@click.option("--video", "video_path", default=None, help="Path to video file (one only)")
@click.option("--system", "system_prompt", default=None, help="System prompt for the agent")
@click.option("--query", default=None, help="Text query")
def main(
    url: str,
    image_paths: tuple[str, ...],
    video_path: str | None,
    system_prompt: str | None,
    query: str | None,
):
    """Run the test client."""
    asyncio.run(_run(url, list(image_paths), video_path, system_prompt, query))


async def _run(
    base_url: str,
    image_paths: list[str],
    video_path: str | None,
    system_prompt: str | None,
    query: str | None,
):
    async with httpx.AsyncClient() as httpx_client:
        # Resolve agent card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        try:
            logger.info(
                "Fetching agent card from %s%s",
                base_url,
                AGENT_CARD_WELL_KNOWN_PATH,
            )
            agent_card = await resolver.get_agent_card()
            logger.info("Agent: %s", agent_card.name)
            logger.info("Description: %s", agent_card.description)
            logger.info(
                "Skills: %s",
                [s.name for s in (agent_card.skills or [])],
            )
        except Exception as e:
            logger.error("Failed to fetch agent card: %s", e)
            sys.exit(1)

        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

        # ── Test 1: Text-only query ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("TEST 1: Text-only query")
        print("=" * 60)

        text_query = query or "What capabilities do you have? List the MCP tools available."
        payload = _build_message(text_query)
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload),
        )

        response = await client.send_message(request)
        print("Response:")
        print(response.model_dump(mode="json", exclude_none=True))

        # ── Test 2: With system prompt ───────────────────────────────────
        if system_prompt:
            print("\n" + "=" * 60)
            print(f"TEST 2: Query with system prompt")
            print(f"  System: {system_prompt}")
            print("=" * 60)

            sp_query = query or "Analyse the provided content."
            sp_payload = _build_message(
                sp_query,
                system_prompt=system_prompt,
                image_paths=image_paths if image_paths else None,
                video_path=video_path,
            )
            sp_request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**sp_payload),
            )

            sp_response = await client.send_message(sp_request)
            print("Response:")
            print(sp_response.model_dump(mode="json", exclude_none=True))

        # ── Test 3: Multiple images ──────────────────────────────────────
        elif image_paths:
            print("\n" + "=" * 60)
            print(f"TEST 2: Multimodal query with {len(image_paths)} image(s)")
            for i, p in enumerate(image_paths):
                print(f"  Image {i + 1}: {p}")
            print("=" * 60)

            mm_query = query or "Describe what you see in these images in detail."
            mm_payload = _build_message(
                mm_query,
                image_paths=image_paths,
            )
            mm_request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**mm_payload),
            )

            mm_response = await client.send_message(mm_request)
            print("Response:")
            print(mm_response.model_dump(mode="json", exclude_none=True))

        # ── Test 4: Video ────────────────────────────────────────────────
        elif video_path:
            print("\n" + "=" * 60)
            print(f"TEST 2: Video query with: {video_path}")
            print("=" * 60)

            vid_query = query or "Describe the key events in this video."
            vid_payload = _build_message(
                vid_query,
                video_path=video_path,
            )
            vid_request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**vid_payload),
            )

            vid_response = await client.send_message(vid_request)
            print("Response:")
            print(vid_response.model_dump(mode="json", exclude_none=True))

        # ── Test 5: Streaming query ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("TEST 3: Streaming query")
        print("=" * 60)

        stream_payload = _build_message(
            "List 3 interesting facts about computer vision."
        )
        stream_request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**stream_payload),
        )

        print("Streaming response:")
        stream_response = client.send_message_streaming(stream_request)
        async for chunk in stream_response:
            print(chunk.model_dump(mode="json", exclude_none=True))

        print("\n" + "=" * 60)
        print("All tests completed.")
        print("=" * 60)


if __name__ == "__main__":
    main()
