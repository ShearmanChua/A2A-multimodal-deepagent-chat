"""
Multimodal DeepAgent — LangGraph DeepAgent with MCP tools and MinIO media upload.

This agent:
1. Accepts text, multiple images, and one video from users via the A2A protocol.
2. Accepts an optional system prompt to customise agent behaviour.
3. Uploads media to MinIO using boto3 and generates pre-signed URLs.
4. Passes the pre-signed URLs to the LLM as both image_url content blocks
   (so the vision model can see them) and as text (so the agent can forward
   the URLs to MCP tools).
5. Connects to an MCP server for tool use (object detection, classification, etc.).
6. Uses DeepAgent for enhanced agent capabilities with middleware and memory.
7. Streams LLM output token-by-token using LangGraph stream_mode="messages".
"""

from __future__ import annotations

import base64
import json as _json
import logging
import os
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any, Literal

import cv2
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# ── DeepAgents ───────────────────────────────────────────────────────────────
try:
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from langchain.agents.middleware import AgentState, before_model
    from langgraph.checkpoint.memory import MemorySaver
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False

# ── Optional MCP client ───────────────────────────────────────────────────────
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from multimodal_agent.minio_uploader import (
    upload_base64,
    upload_file,
)

logger = logging.getLogger(__name__)

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://research-phoenix-1:6006/v1/traces"

tracer_provider = register(
    endpoint="http://research-phoenix-1:6006/v1/traces",
    project_name="mcp-test"
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://research-mcp-server-1:8000")
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "8"))

# Image mode: "base64" = pass images directly as base64 data URLs to LLM
#             "minio" = upload to MinIO and use pre-signed URLs
# Default to "base64" since OpenAI cannot access internal MinIO URLs
IMAGE_MODE = os.environ.get("IMAGE_MODE", "base64").lower()

# Memories directory for DeepAgent
MEMORIES_DIR = Path(os.environ.get("MEMORIES_DIR", "/app/src/memories"))
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------


def _sample_video_frames_b64(video_path: str, n_frames: int = MAX_VIDEO_FRAMES) -> list[str]:
    """Sample ``n_frames`` evenly-spaced frames from a video.

    Returns a list of raw base64 strings (JPEG encoded, no data-URL prefix).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames: list[str] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode())

    cap.release()
    return frames


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# DeepAgent middleware - extract images from tool messages
# ---------------------------------------------------------------------------


async def extract_images_to_human(messages: list) -> list:
    """Convert MCP ToolMessage ImageContent blocks into a HumanMessage.
    
    This middleware processes tool outputs that contain images and converts
    them into a format the LLM can understand.
    """
    new_messages: list = []
    collected_images: list[dict] = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            images_found = [
                item for item in msg.content
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            if images_found:
                msg.content = "images retrieved"
                collected_images.extend(images_found)
        new_messages.append(msg)

    if collected_images:
        new_messages.append(AIMessage(content="(images retrieved)"))
        new_messages.append(HumanMessage(content=collected_images))

    return new_messages


if DEEPAGENTS_AVAILABLE:
    @before_model
    async def parse_messages_before_model(
        state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Middleware that runs before each model call to process messages."""
        state["messages"] = await extract_images_to_human(state["messages"])
        return state


# ---------------------------------------------------------------------------
# Tool error-handling wrapper
# ---------------------------------------------------------------------------


def _wrap_tools_with_error_handling(tools: list) -> list:
    """Wrap every tool so exceptions become friendly error strings for the LLM.
    
    For MCP tools (StructuredTool), we only wrap the async method since they
    don't support sync invocation.
    """
    from functools import wraps

    def _error_formatter(exc: ToolException) -> str:
        return (
            f"⚠️ Tool error: {exc}\n"
            "You may retry with corrected parameters, try an alternative tool, "
            "or skip this step."
        )

    for tool in tools:
        tool.handle_tool_error = _error_formatter

        # Check if this is an async-only tool (like MCP StructuredTool)
        # by checking if _run raises NotImplementedError
        is_async_only = False
        try:
            # Check the tool's coroutine attribute or if it's a StructuredTool
            if hasattr(tool, 'coroutine') and tool.coroutine is not None:
                is_async_only = True
        except Exception:
            pass

        # Only wrap _run if it's not an async-only tool
        if not is_async_only and hasattr(tool, "_run"):
            original_run = tool._run

            def _make_safe_run(orig):
                @wraps(orig)
                def _safe_run(*args, **kwargs):
                    try:
                        return orig(*args, **kwargs)
                    except ToolException:
                        raise
                    except NotImplementedError:
                        # This is an async-only tool, re-raise
                        raise
                    except Exception as exc:
                        raise ToolException(f"{type(exc).__name__}: {exc}") from exc
                return _safe_run

            tool._run = _make_safe_run(original_run)

        if hasattr(tool, "_arun"):
            original_arun = tool._arun

            def _make_safe_arun(orig):
                @wraps(orig)
                async def _safe_arun(*args, **kwargs):
                    try:
                        return await orig(*args, **kwargs)
                    except ToolException:
                        raise
                    except Exception as exc:
                        raise ToolException(f"{type(exc).__name__}: {exc}") from exc
                return _safe_arun

            tool._arun = _make_safe_arun(original_arun)

    return tools


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------


class ResponseFormat(BaseModel):
    """Structured response from the agent."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


# ---------------------------------------------------------------------------
# MultimodalAgent
# ---------------------------------------------------------------------------


class MultimodalAgent:
    """A2A-compatible multimodal agent that uploads media to MinIO and uses
    pre-signed URLs for both the vision model and MCP tools.

    Uses DeepAgent for enhanced capabilities including:
    - Middleware for processing messages before model calls
    - Memory files for persistent agent state
    - Composite backends for flexible storage

    Supports:
    - An optional system prompt to customise behaviour per request.
    - Multiple images (each uploaded separately to MinIO).
    - One video (frames sampled and uploaded to MinIO).
    """

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a helpful multimodal research assistant powered by LangGraph DeepAgent.\n\n"
        "Guidelines:\n"
        "- Use the `write_todos` tool at the start of every multi-step task.\n"
        "- When images or videos are provided, analyse them carefully before responding.\n"
        "- When calling MCP tools that require an image, pass the base64 data or URL "
        "  provided in the message.\n"
        "- Be concise but thorough.\n"
        "- If you need more information from the user, ask for it.\n"
    )

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide "
        "more information.\n"
        "Set response status to error if there is an error.\n"
        "Set response status to completed if the request is complete.\n"
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "image/png", "image/jpeg", "video/mp4"]

    def __init__(self):
        model_source = os.getenv("model_source", "openai")

        if model_source == "openai":
            self.model = ChatOpenAI(
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                base_url=os.getenv("MODEL_ENDPOINT") or None,
                api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
                temperature=0,
            )
        else:
            self.model = ChatOpenAI(
                model=os.getenv("TOOL_LLM_NAME", "gpt-4o"),
                openai_api_key=os.getenv("API_KEY", "EMPTY"),
                openai_api_base=os.getenv("TOOL_LLM_URL"),
                temperature=0,
            )

        self.tools: list = []
        self.mcp_client: MultiServerMCPClient | None = None
        self.agent = None
        self._initialized = False
        # Checkpointer for conversation memory (persists across requests)
        self._checkpointer = MemorySaver() if DEEPAGENTS_AVAILABLE else None

    async def _ensure_initialized(self) -> None:
        """Lazily initialize MCP tools and the DeepAgent."""
        if self._initialized:
            return

        # Connect to MCP server for tools
        if MCP_AVAILABLE and MCP_SERVER_URL:
            try:
                self.mcp_client = MultiServerMCPClient(
                    {"mcp": {"transport": "http", "url": f"{MCP_SERVER_URL}/mcp"}}
                )
                self.tools = await self.mcp_client.get_tools()
                logger.info("Connected to MCP server – %d tool(s) available", len(self.tools))
            except Exception as exc:
                logger.warning("MCP tools unavailable: %s", exc)
                self.mcp_client = None

        if self.tools:
            self.tools = _wrap_tools_with_error_handling(self.tools)
            logger.info("%d MCP tool(s) wrapped with error handling", len(self.tools))

        # Build the agent
        if DEEPAGENTS_AVAILABLE:
            # Use DeepAgent with middleware, memory, and checkpointer for conversation continuity
            middleware = [parse_messages_before_model]
            
            self.agent = create_deep_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=self.DEFAULT_SYSTEM_INSTRUCTION,
                middleware=middleware,
                memory=["/memories/AGENTS.md"],
                checkpointer=self._checkpointer,  # Enable conversation memory
                backend=CompositeBackend(
                    default=StateBackend(),
                    routes={
                        "/memories/": FilesystemBackend(
                            root_dir=str(MEMORIES_DIR),
                            virtual_mode=True,
                        ),
                    },
                ),
            )
            logger.info("DeepAgent initialized with middleware, memory, and checkpointer")
        else:
            # Fallback to basic LangGraph agent
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            
            memory = MemorySaver()
            self.agent = create_react_agent(
                self.model,
                tools=self.tools,
                checkpointer=memory,
                prompt=self.DEFAULT_SYSTEM_INSTRUCTION,
                response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
            )
            logger.info("Fallback to basic LangGraph agent (DeepAgents not available)")

        self._initialized = True

    # ------------------------------------------------------------------
    # Media processing: upload to MinIO, get pre-signed URLs
    # ------------------------------------------------------------------

    def _upload_image_file(self, image_path: str) -> str:
        """Upload a single image file to MinIO and return its pre-signed URL."""
        url = upload_file(image_path, prefix="a2a/images")
        logger.debug("Uploaded image to MinIO: %s", url[:80])
        return url

    def _upload_image_b64(self, b64_data: str) -> str:
        """Upload a single base64-encoded image to MinIO and return its pre-signed URL."""
        url = upload_base64(b64_data, ext="jpg", prefix="a2a/images")
        logger.debug("Uploaded base64 image to MinIO: %s", url[:80])
        return url

    def _upload_video_file(self, video_path: str) -> list[str]:
        """Upload video frames to MinIO and return pre-signed URLs for each frame."""
        frames_b64 = _sample_video_frames_b64(video_path)
        if not frames_b64:
            return []

        urls: list[str] = []
        for i, frame_b64 in enumerate(frames_b64):
            url = upload_base64(frame_b64, ext="jpg", prefix="a2a/video_frames")
            urls.append(url)

        logger.debug("Uploaded %d video frames to MinIO", len(urls))
        return urls

    def _upload_video_b64(self, b64_data: str) -> str:
        """Upload a base64-encoded video to MinIO and return its pre-signed URL."""
        url = upload_base64(b64_data, ext="mp4", prefix="a2a/videos")
        logger.debug("Uploaded base64 video to MinIO: %s", url[:80])
        return url

    # ------------------------------------------------------------------
    # Build LangChain message content with images
    # ------------------------------------------------------------------

    def _build_message_content_base64(
        self,
        text: str,
        image_b64s: list[str] | None = None,
        video_frame_b64s: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> list[dict] | str:
        """Build ``content`` for a HumanMessage using base64 data URLs.

        This mode passes images directly to the LLM as base64 data URLs,
        which works with OpenAI and other providers that cannot access
        internal MinIO URLs.

        If image_urls are also provided (for MCP tools), they are included
        in the text portion of the message.
        """
        has_images = image_b64s and len(image_b64s) > 0
        has_video = video_frame_b64s and len(video_frame_b64s) > 0

        if not has_images and not has_video:
            return text

        # Build text that includes URLs for MCP tools (if available)
        url_text = text
        
        if image_urls and len(image_urls) > 0:
            url_text += "\n\n"
            url_text += f"[{len(image_urls)} image(s) uploaded to MinIO. Pre-signed URLs for MCP tools:]\n"
            for i, url in enumerate(image_urls):
                url_text += f"  Image {i + 1}: {url}\n"
            url_text += "Pass these URLs to any MCP tool that requires image input.\n"

        blocks: list[dict] = [{"type": "text", "text": url_text}]

        # Add image_url blocks with base64 data URLs for the vision model
        if has_images:
            for i, b64 in enumerate(image_b64s):
                blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
                # Create data URL from base64
                data_url = f"data:image/jpeg;base64,{b64}"
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })

        if has_video:
            blocks.append({"type": "text", "text": "[Video frames]"})
            for b64 in video_frame_b64s:
                data_url = f"data:image/jpeg;base64,{b64}"
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })

        return blocks

    def _build_message_content_minio(
        self,
        text: str,
        image_urls: list[str] | None = None,
        video_frame_urls: list[str] | None = None,
    ) -> list[dict] | str:
        """Build ``content`` for a HumanMessage using MinIO pre-signed URLs.

        This mode uploads images to MinIO and uses pre-signed URLs.
        Only works if the LLM can access the MinIO endpoint (e.g., local models).
        """
        has_images = image_urls and len(image_urls) > 0
        has_video = video_frame_urls and len(video_frame_urls) > 0

        if not has_images and not has_video:
            return text

        # Build text that includes the URLs so the agent can pass them to tools
        url_text = text + "\n\n"

        if has_images:
            url_text += f"[{len(image_urls)} image(s) uploaded to MinIO. Pre-signed URLs:]\n"
            for i, url in enumerate(image_urls):
                url_text += f"  Image {i + 1}: {url}\n"
            url_text += (
                "Pass these URLs to any MCP tool that requires image input.\n\n"
            )

        if has_video:
            url_text += (
                f"[Video uploaded — {len(video_frame_urls)} frame(s) extracted and "
                "uploaded to MinIO. Pre-signed URLs for each frame:]\n"
            )
            for i, url in enumerate(video_frame_urls):
                url_text += f"  Frame {i}: {url}\n"
            url_text += (
                "Pass these URLs to any MCP tool that requires image input. "
                "Each URL is a direct link to the frame image.\n"
            )

        blocks: list[dict] = [{"type": "text", "text": url_text}]

        # Add image_url blocks so the vision model can see the images
        if has_images:
            for i, url in enumerate(image_urls):
                blocks.append({"type": "text", "text": f"[Image {i + 1}]"})
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

        if has_video:
            blocks.append({"type": "text", "text": "[Video frames]"})
            for url in video_frame_urls:
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

        return blocks

    # ------------------------------------------------------------------
    # Streaming interface (A2A compatible)
    # ------------------------------------------------------------------

    async def stream(
        self,
        query: str,
        context_id: str,
        system_prompt: str | None = None,
        image_paths: list[str] | None = None,
        image_b64s: list[str] | None = None,
        video_path: str | None = None,
        video_b64: str | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Process a user query with optional system prompt and media.

        Parameters
        ----------
        query:
            The user's text query.
        context_id:
            Conversation thread ID.
        system_prompt:
            Optional system prompt to customise agent behaviour for this
            request. If not provided, the default system instruction is used.
        image_paths:
            Optional list of local file paths to images (multiple allowed).
        image_b64s:
            Optional list of base64-encoded image data (multiple allowed).
        video_path:
            Optional local file path to a single video.
        video_b64:
            Optional base64-encoded video data (single video only).

        Yields
        ------
        dict with ``is_task_complete``, ``require_user_input``, ``content``.
        """
        await self._ensure_initialized()

        # Collect all base64 images (from paths or direct b64)
        all_image_b64s: list[str] = []
        image_urls: list[str] = []
        video_frame_b64s: list[str] = []
        video_frame_urls: list[str] = []

        # -- Process image paths --
        if image_paths:

            for path in image_paths:
                # Read file and convert to base64
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                all_image_b64s.append(b64)
                
                # Also upload to MinIO for MCP tools
                if IMAGE_MODE == "minio":
                    url = self._upload_image_file(path)
                    image_urls.append(url)

        # -- Process base64 images --
        if image_b64s:

            for b64 in image_b64s:
                all_image_b64s.append(b64)
                
                # Also upload to MinIO for MCP tools
                url = self._upload_image_b64(b64)
                image_urls.append(url)

        # -- Process video --
        if video_path:

            # Sample frames as base64
            video_frame_b64s = _sample_video_frames_b64(video_path)
            
            # Also upload frames to MinIO for MCP tools
            for frame_b64 in video_frame_b64s:
                url = upload_base64(frame_b64, ext="jpg", prefix="a2a/video_frames")
                video_frame_urls.append(url)

        elif video_b64:

            # Upload the raw video file to MinIO
            video_url = self._upload_video_b64(video_b64)
            video_frame_urls = [video_url]
            # Note: For base64 video, we can't easily extract frames without saving to disk
            # So we just use the video URL for MCP tools

        # -- Build the message content based on IMAGE_MODE --
        if IMAGE_MODE == "base64":
            # Use base64 data URLs for the LLM, but include MinIO URLs for MCP tools
            content = self._build_message_content_base64(
                query,
                image_b64s=all_image_b64s if all_image_b64s else None,
                video_frame_b64s=video_frame_b64s if video_frame_b64s else None,
                image_urls=image_urls if image_urls else None,
            )
        else:
            # Use MinIO URLs for both LLM and MCP tools
            content = self._build_message_content_minio(
                query,
                image_urls=image_urls if image_urls else None,
                video_frame_urls=video_frame_urls if video_frame_urls else None,
            )

        # -- Assemble messages --
        messages: list = []

        # Add system prompt if provided (overrides default for this turn)
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=content))

        input_payload = {"messages": messages}
        config = {"configurable": {"thread_id": context_id}, "recursion_limit": 100}

        # Stream through the agent using stream_mode="values" only.
        # This gives us complete state snapshots after each node completes.
        #
        # For AIMessage with tool_calls: emit "thought" event with content (once)
        # For ToolMessage: emit "tool_end" event
        # For final AIMessage (no tool_calls): simulate streaming by yielding
        #   content in chunks as "token" events, then "final" event.
        #
        # Reference: https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/

        # Track processed message IDs to avoid duplicates from subgraphs
        processed_msg_ids: set[str] = set()

        async def _process_values_stream(data: dict):
            """Process a values stream snapshot and yield appropriate events."""
            import asyncio
            
            message = data["messages"][-1]
            msg_type = type(message).__name__
            msg_id = getattr(message, "id", None)
            
            # Skip if we've already processed this message
            if msg_id and msg_id in processed_msg_ids:
                logger.debug("[stream] Skipping duplicate msg_id=%s", msg_id)
                return
            if msg_id:
                processed_msg_ids.add(msg_id)
            
            has_tool_calls = (
                hasattr(message, "tool_calls")
                and message.tool_calls
                and len(message.tool_calls) > 0
            )
            logger.info(
                "[stream] values: last_msg_type=%s, "
                "is_AIMessage=%s, has_tool_calls=%s, has_content=%s, "
                "total_messages=%d, msg_id=%s",
                msg_type,
                isinstance(message, AIMessage),
                has_tool_calls,
                bool(getattr(message, "content", None)),
                len(data["messages"]),
                msg_id,
            )

            # --- AIMessage with tool_calls: emit thought + tool_start ---
            # Note: We check for tool_calls to distinguish between intermediate
            # steps (with tool_calls) and final response (without tool_calls).
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                logger.info(
                    "[stream] AIMessage with %d tool_calls",
                    len(message.tool_calls),
                )
                
                # Emit the thought (LLM reasoning) if present
                if message.content:
                    logger.info(
                        "[stream] Emitting thought, content_len=%d",
                        len(message.content),
                    )
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": message.content,
                        "event_type": "thought",
                    }

                # Emit tool_start for each tool call
                for tc in message.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    try:
                        input_str = _json.dumps(tool_args, indent=2)
                    except Exception:
                        input_str = str(tool_args)
                    if len(input_str) > 500:
                        input_str = input_str[:500] + "…"
                    logger.info("[stream] Emitting tool_start: %s", tool_name)
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"🔧 Calling tool: **{tool_name}**",
                        "event_type": "tool_start",
                        "tool_name": tool_name,
                        "tool_input": input_str,
                    }

            # --- ToolMessage: emit tool_end ---
            elif isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", "unknown")
                tool_output = message.content
                output_str = str(tool_output)
                has_media = (
                    "'type': 'image'" in output_str or
                    '"type": "image"' in output_str or
                    "'base64':" in output_str or
                    '"base64":' in output_str or
                    "data:image/" in output_str
                )
                if not has_media and len(output_str) > 800:
                    output_str = output_str[:800] + "…"
                logger.info("[stream] Emitting tool_end: %s", tool_name)
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"✅ Tool **{tool_name}** completed",
                    "event_type": "tool_end",
                    "tool_name": tool_name,
                    "tool_output": output_str,
                }

            # --- Final AIMessage (no tool_calls): simulate streaming ---
            # Note: We accept both AIMessage and AIMessageChunk here because
            # the final response might come as a chunk in some streaming modes.
            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and message.content
            ):
                content = message.content
                logger.info(
                    "[stream] Final response, simulating streaming, content_len=%d",
                    len(content),
                )
                
                # Simulate streaming by yielding content in chunks
                # Use word-based chunking for more natural streaming
                words = content.split(' ')
                chunk_size = 3  # words per chunk
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk = ' '.join(chunk_words)
                    # Add space back except for last chunk
                    if i + chunk_size < len(words):
                        chunk += ' '
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": chunk,
                        "event_type": "token",
                    }
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.01)
                
                # Emit final event with complete content
                logger.info("[stream] Emitting final")
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": content,
                    "event_type": "final_response",
                }

            else:
                logger.debug(
                    "[stream] Unhandled values message: type=%s, "
                    "content_preview=%s",
                    msg_type,
                    str(getattr(message, "content", ""))[:100],
                )

        # Stream using values mode only
        logger.info("[stream] Starting astream with stream_mode='values'")
        async for namespace, data in self.agent.astream(
            input_payload, config, stream_mode="values", subgraphs=True
        ):
            logger.debug("[stream] namespace=%s", namespace)
            async for event in _process_values_stream(data):
                yield event

    def _get_agent_response(self, config: dict) -> dict[str, Any]:
        """Extract the structured response from the agent's final state."""
        current_state = self.agent.get_state(config)
        structured_response = current_state.values.get("structured_response")

        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "Unable to process your request at the moment. "
                "Please try again."
            ),
        }
