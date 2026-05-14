"""Multimodal DeepAgent — LangGraph agent with MCP tools and SeaweedFS media upload."""

from __future__ import annotations

import base64
import json as _json
import logging
import os
from collections.abc import AsyncIterable
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from multimodal_agent.configs.config import IMAGE_MODE, MEMORIES_DIR, MCP_SERVER_URL, SKILLS_DIR, UPLOADS_DIR, object_store_available
from multimodal_agent.agent.content_builders import build_content_base64, build_content_object_store
from multimodal_agent.agent.middleware import parse_messages_before_model
from multimodal_agent.utils.object_store_uploader import upload_base64, upload_file, upload_text_to_key
from multimodal_agent.agent.tools import ResponseFormat, wrap_tools_with_error_handling
from multimodal_agent.backends.object_store_backend import ObjectStoreBackend

_UPLOADS_BUCKET = os.environ.get("OBJECT_STORE_UPLOADS_BUCKET", "uploads")

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


def _extract_text(content: Any) -> str:
    """Extract plain text from message content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""



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
        " - If the retrieved chunks do not fully answer the question, or there is another "
        "relevant collection,  perform query expansionand query_weaviate again.\n"
        " - If the retrieved chunks contain images, call `get_object_store_image_base64` "
        "  to retrieve them and analyse them.\n"
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
            if object_store_available():
                uploads_backend = ObjectStoreBackend(bucket=_UPLOADS_BUCKET)
                logger.info("Using ObjectStoreBackend for /uploads/ (bucket: %s)", _UPLOADS_BUCKET)
            else:
                uploads_backend = FilesystemBackend(root_dir=str(UPLOADS_DIR), virtual_mode=True)
                logger.info("Using FilesystemBackend for /uploads/ (dir: %s)", UPLOADS_DIR)

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
                        "/uploads/": uploads_backend,
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
        documents: list[tuple[str, str]] | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream agent responses for a user query with optional media.

        Yields dicts with keys: ``is_task_complete``, ``require_user_input``,
        ``content``, ``event_type`` (token | thought | tool_start | tool_end |
        final_response).
        """
        await self._ensure_initialized()

        # Pre-seed the /uploads/<context_id>/ virtual directory so the agent can
        # read documents immediately without calling write_file first.
        if documents:
            file_list = "\n".join(
                f"  • /uploads/{context_id}/{name}" for name, _ in documents
            )
            if object_store_available():
                for name, text in documents:
                    key = f"{context_id}/{name}"
                    upload_text_to_key(text, key, bucket=_UPLOADS_BUCKET)
                    logger.info("Uploaded document to s3://%s/%s", _UPLOADS_BUCKET, key)
            else:
                upload_subdir = UPLOADS_DIR / context_id
                upload_subdir.mkdir(parents=True, exist_ok=True)
                for name, text in documents:
                    (upload_subdir / name).write_text(text, encoding="utf-8")
                    logger.info("Wrote document to %s/%s/%s", UPLOADS_DIR, context_id, name)

            query = (
                f"The user has uploaded {len(documents)} document(s). "
                f"They are already available on the virtual filesystem:\n{file_list}\n"
                f"Use read_file, grep, or glob to access them.\n\n"
            ) + query

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
        logger.info("[stream] starting — context_id=%s query_len=%d", context_id, len(query))
        logger.debug("[stream] input_payload: %s", str(input_payload)[:500])
        config = {"configurable": {"thread_id": context_id}, "recursion_limit": 100}

        # ── Per-LLM-invocation streaming state ─────────────────────────────
        # Each LLM call produces chunks with the same message ID.  When the ID
        # changes we know a new invocation has started (e.g. after tool results).
        current_msg_id: str | None = None
        turn_has_thought = False   # True once we've emitted a "thought" for this turn
        turn_content = ""          # Accumulated content for the current LLM turn
        processed_update_ids: set[str] = set()  # Dedup complete messages from updates
        # Guard against middleware nodes whose `updates` AIMessage is not the real
        # final response.  Middleware (e.g. parse_messages_before_model.before_model)
        # never produces `messages` stream chunks, so this flag stays False during
        # their updates and we skip them.
        has_pending_llm_output = False

        async for chunk in self.agent.astream(
            input_payload, config,
            stream_mode=["messages", "updates"],
            version="v2",
            subgraphs=True,
        ):
            ctype = chunk["type"]
            data = chunk["data"]

            # ── Real-time LLM token streaming ─────────────────────────────
            if ctype == "messages":
                msg_chunk, meta = data
                if not isinstance(msg_chunk, AIMessageChunk):
                    continue

                node = meta.get("langgraph_node", "?")
                msg_id = getattr(msg_chunk, "id", None)

                # Detect new LLM invocation → reset per-turn state
                if msg_id and msg_id != current_msg_id:
                    logger.debug("[stream] new LLM turn id=%s node=%s", msg_id, node)
                    current_msg_id = msg_id
                    turn_has_thought = False
                    turn_content = ""

                chunk_text = _extract_text(msg_chunk.content)
                if chunk_text:
                    turn_content += chunk_text
                    has_pending_llm_output = True
                    logger.debug("[stream] token node=%s len=%d: %r", node, len(chunk_text), chunk_text[:60])
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": chunk_text,
                        "event_type": "token",
                    }

                # First tool-call chunk → retroactively label accumulated content as a thought
                if msg_chunk.tool_call_chunks:
                    has_pending_llm_output = True

                if msg_chunk.tool_call_chunks and not turn_has_thought:
                    turn_has_thought = True
                    logger.info(
                        "[stream] thought signalled turn=%s content_len=%d: %r",
                        current_msg_id, len(turn_content), turn_content[:120],
                    )
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": turn_content,
                        "event_type": "thought",
                    }

            # ── Node-completion events (tool calls, tool results, final response) ─
            elif ctype == "updates":
                if not isinstance(data, dict):
                    continue

                for node_name, node_data in data.items():
                    if not isinstance(node_data, dict):
                        continue

                    msgs = node_data.get("messages", [])
                    if not isinstance(msgs, list):
                        msgs = [msgs]

                    for msg in msgs:
                        uid = getattr(msg, "id", None)
                        if uid:
                            if uid in processed_update_ids:
                                logger.debug("[stream] skipping duplicate update msg id=%s", uid)
                                continue
                            processed_update_ids.add(uid)

                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            if not has_pending_llm_output:
                                logger.debug(
                                    "[stream] skip tool-call AIMessage — no preceding messages chunks (node=%s)",
                                    node_name,
                                )
                                continue
                            has_pending_llm_output = False
                            logger.info(
                                "[stream] tool_calls node=%s tools=%s",
                                node_name, [tc.get("name") for tc in msg.tool_calls],
                            )
                            for tc in msg.tool_calls:
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

                        elif isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, "name", "unknown")
                            output_str = str(msg.content)
                            has_media = any(
                                m in output_str
                                for m in ('"type": "image"', "'type': 'image'", "data:image/")
                            )
                            if not has_media and len(output_str) > 800:
                                output_str = output_str[:800] + "…"
                            logger.info("[stream] tool_end node=%s tool=%s", node_name, tool_name)
                            yield {
                                "is_task_complete": False,
                                "require_user_input": False,
                                "content": f"✅ Tool **{tool_name}** completed",
                                "event_type": "tool_end",
                                "tool_name": tool_name,
                                "tool_output": output_str,
                            }

                        elif isinstance(msg, AIMessage) and not msg.tool_calls:
                            if not has_pending_llm_output:
                                logger.debug(
                                    "[stream] skip AIMessage — no preceding messages chunks, likely middleware (node=%s content=%r)",
                                    node_name, _extract_text(msg.content)[:80],
                                )
                                continue
                            has_pending_llm_output = False
                            text = _extract_text(msg.content)
                            if text:
                                logger.info(
                                    "[stream] final_response node=%s len=%d: %r",
                                    node_name, len(text), text[:120],
                                )
                                yield {
                                    "is_task_complete": True,
                                    "require_user_input": False,
                                    "content": text,
                                    "event_type": "final_response",
                                }

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
