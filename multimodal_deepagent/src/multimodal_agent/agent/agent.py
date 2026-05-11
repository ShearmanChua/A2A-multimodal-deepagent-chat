"""Multimodal DeepAgent — LangGraph agent with MCP tools and SeaweedFS media upload."""

from __future__ import annotations

import asyncio
import base64
import json as _json
import logging
import os
from collections.abc import AsyncIterable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from multimodal_agent.configs.config import IMAGE_MODE, MEMORIES_DIR, MCP_SERVER_URL, SKILLS_DIR, object_store_available
from multimodal_agent.agent.content_builders import build_content_base64, build_content_object_store
from multimodal_agent.agent.middleware import parse_messages_before_model
from multimodal_agent.utils.object_store_uploader import upload_base64, upload_file
from multimodal_agent.agent.tools import ResponseFormat, wrap_tools_with_error_handling

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from langgraph.checkpoint.memory import MemorySaver
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ── Tracing (optional) ────────────────────────────────────────────────────────

if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
    _tracer = register(
        endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "RAG-agent"),
    )
    LangChainInstrumentor().instrument(tracer_provider=_tracer)

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "image/png", "image/jpeg", "video/mp4"]


# ---------------------------------------------------------------------------
# MultimodalAgent
# ---------------------------------------------------------------------------


class MultimodalAgent:
    """A2A-compatible multimodal agent with SeaweedFS media upload and MCP tools.

    Supports multiple images, one video, and an optional per-request system prompt.
    Uses DeepAgent when available, otherwise falls back to a basic react agent.
    """

    SUPPORTED_CONTENT_TYPES = SUPPORTED_CONTENT_TYPES

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a helpful multimodal RAG assistant.\n\n"
        "## Answering queries\n"
        "- **Always search the knowledge base first** before answering any factual question.\n"
        "  1. Call `list_weaviate_collections` to see what collections exist.\n"
        "  2. Call `query_weaviate` with the user's question on the most relevant collection(s).\n"
        "  3. Synthesise the retrieved chunks into a clear answer and cite sources.\n"
        "- If the retrieved chunks do not fully answer the question, perform query expansion"
        " and query_weaviate again.\n"
        "- If the knowledge base returns no useful results, say so and answer from general knowledge"
        "  only when you are confident — otherwise ask the user to ingest the relevant documents.\n\n"
        "## General guidelines\n"
        "- Use the `write_todos` tool at the start of every multi-step task.\n"
        "- When images or videos are provided, analyse them carefully before responding.\n"
        "- When calling MCP tools that require an image, pass the URL provided in the message.\n"
        "- Be concise but thorough.\n"
        "- If you need more information from the user, ask for it.\n"
    )

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information.\n"
        "Set response status to error if there is an error.\n"
        "Set response status to completed if the request is complete.\n"
    )

    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            base_url=os.getenv("MODEL_ENDPOINT") or None,
            api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
            temperature=0,
        )
        self.tools: list = []
        self.mcp_client: MultiServerMCPClient | None = None
        self.agent = None
        self._initialized = False
        self._checkpointer = MemorySaver() if DEEPAGENTS_AVAILABLE else None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Lazily connect to MCP and build the agent graph (runs once)."""
        if self._initialized:
            return

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
            self.tools = wrap_tools_with_error_handling(self.tools)

        if DEEPAGENTS_AVAILABLE:
            self.agent = create_deep_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=self.DEFAULT_SYSTEM_INSTRUCTION,
                middleware=[parse_messages_before_model],
                memory=["/memories/AGENTS.md"],
                skills=["/skills/"],
                checkpointer=self._checkpointer,
                backend=CompositeBackend(
                    default=StateBackend(),
                    routes={
                        "/memories/": FilesystemBackend(
                            root_dir=str(MEMORIES_DIR),
                            virtual_mode=True,
                        ),
                        "/skills/": FilesystemBackend(
                            root_dir=str(SKILLS_DIR),
                            virtual_mode=True,
                        ),
                    },
                ),
            )
            logger.info("DeepAgent initialized")
        else:
            from langgraph.checkpoint.memory import MemorySaver as _MemorySaver
            from langgraph.prebuilt import create_react_agent

            self.agent = create_react_agent(
                self.model,
                tools=self.tools,
                checkpointer=_MemorySaver(),
                prompt=self.DEFAULT_SYSTEM_INSTRUCTION,
                response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
            )
            logger.info("Fallback react agent initialized (DeepAgents not available)")

        self._initialized = True

    # ------------------------------------------------------------------
    # Media upload helpers
    # ------------------------------------------------------------------

    def _upload_image_file(self, path: str) -> str:
        return upload_file(path, prefix="a2a/images")

    def _upload_image_b64(self, b64_data: str) -> str:
        return upload_base64(b64_data, ext="jpg", prefix="a2a/images")

    def _upload_video_b64(self, b64_data: str) -> str:
        return upload_base64(b64_data, ext="mp4", prefix="a2a/videos")

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
        image_urls: list[str] | None = None,
        video_path: str | None = None,
        video_b64: str | None = None,
        video_urls: list[str] | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream agent responses for a user query with optional media.

        Yields dicts with keys: ``is_task_complete``, ``require_user_input``,
        ``content``, ``event_type`` (token | thought | tool_start | tool_end |
        final_response).
        """
        await self._ensure_initialized()

        all_image_b64s: list[str] = []
        uploaded_image_urls: list[str] = []
        uploaded_video_urls: list[str] = []
        inline_video_b64: str | None = None
        swfs = object_store_available()

        for path in image_paths or []:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            all_image_b64s.append(b64)
            if swfs:
                uploaded_image_urls.append(self._upload_image_file(path))

        for b64 in image_b64s or []:
            all_image_b64s.append(b64)
            if swfs:
                uploaded_image_urls.append(self._upload_image_b64(b64))

        if video_path:
            with open(video_path, "rb") as f:
                inline_video_b64 = base64.b64encode(f.read()).decode()
            if swfs:
                uploaded_video_urls.append(upload_file(video_path, prefix="a2a/videos"))
        elif video_b64:
            inline_video_b64 = video_b64
            if swfs:
                uploaded_video_urls.append(self._upload_video_b64(video_b64))

        all_image_urls = uploaded_image_urls + list(image_urls or [])
        all_video_urls = uploaded_video_urls + list(video_urls or [])

        use_swfs_urls = IMAGE_MODE == "object_store" and swfs
        if use_swfs_urls:
            content = build_content_object_store(
                query,
                image_urls=all_image_urls or None,
                video_urls=all_video_urls or None,
            )
        else:
            content = build_content_base64(
                query,
                image_b64s=all_image_b64s or None,
                image_urls=all_image_urls or None,
                video_b64=inline_video_b64,
                video_urls=all_video_urls or None,
            )

        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=content))

        input_payload = {"messages": messages}
        config = {"configurable": {"thread_id": context_id}, "recursion_limit": 100}
        processed_ids: set[str] = set()

        # Process the snapshot
        async def _process_snapshot(data: dict):
            message = data["messages"][-1]
            msg_id = getattr(message, "id", None)
            if msg_id:
                if msg_id in processed_ids:
                    return
                processed_ids.add(msg_id)

            if isinstance(message, AIMessage) and message.tool_calls:
                if message.content:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": message.content,
                        "event_type": "thought",
                    }
                for tc in message.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    try:
                        input_str = _json.dumps(tc.get("args", {}), indent=2)
                    except Exception:
                        input_str = str(tc.get("args", {}))
                    if len(input_str) > 500:
                        input_str = input_str[:500] + "…"
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"🔧 Calling tool: **{tool_name}**",
                        "event_type": "tool_start",
                        "tool_name": tool_name,
                        "tool_input": input_str,
                    }

            elif isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", "unknown")
                output_str = str(message.content)
                has_media = any(
                    marker in output_str
                    for marker in ('"type": "image"', "'type': 'image'", "data:image/")
                )
                if not has_media and len(output_str) > 800:
                    output_str = output_str[:800] + "…"
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"✅ Tool **{tool_name}** completed",
                    "event_type": "tool_end",
                    "tool_name": tool_name,
                    "tool_output": output_str,
                }

            elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
                text = message.content
                words = text.split(" ")
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i + 3])
                    if i + 3 < len(words):
                        chunk += " "
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": chunk,
                        "event_type": "token",
                    }
                    await asyncio.sleep(0.01)
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": text,
                    "event_type": "final_response",
                }

        # Run the agent
        async for _, data in self.agent.astream(
            input_payload, config, stream_mode="values", subgraphs=True
        ):
            async for event in _process_snapshot(data):
                yield event

    # ------------------------------------------------------------------
    # State helper
    # ------------------------------------------------------------------

    def _get_agent_response(self, config: dict) -> dict[str, Any]:
        """Extract the structured response from the agent's final state."""
        state = self.agent.get_state(config)
        resp = state.values.get("structured_response")
        if resp and isinstance(resp, ResponseFormat):
            if resp.status == "input_required":
                return {"is_task_complete": False, "require_user_input": True, "content": resp.message}
            if resp.status == "error":
                return {"is_task_complete": False, "require_user_input": True, "content": resp.message}
            if resp.status == "completed":
                return {"is_task_complete": True, "require_user_input": False, "content": resp.message}
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "Unable to process your request at the moment. Please try again.",
        }
