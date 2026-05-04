"""
A2A AgentExecutor for the Multimodal DeepAgent.

Handles incoming A2A requests, extracts text, system prompt, multiple images,
and one video from message parts, delegates to the MultimodalAgent, and
streams status updates / artifacts back.

Each streaming event carries structured ``metadata`` on both the
:class:`~a2a.types.Message` and the :pymethod:`TaskUpdater.update_status`
call so that downstream consumers (frontends, orchestrators) can
distinguish event kinds without parsing text prefixes.

Metadata schema (``message.metadata`` / ``update_status`` *metadata*):

* **token**          – ``{"event_type": "token"}`` (simulated streaming chunks)
* **tool_call**      – ``{"event_type": "tool_call", "tool_name": "…", "tool_input": "…"}``
* **llm_thought**    – ``{"event_type": "llm_thought"}`` (LLM reasoning before tool calls)
* **tool_result**    – ``{"event_type": "tool_result", "tool_name": "…", "tool_output": "…"}``
* **final_response** – ``{"event_type": "final_response"}``
* **status**         – ``{"event_type": "status"}``  (generic progress)
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskStore, TaskUpdater
from a2a.types import (
    FilePart,
    InternalError,
    InvalidParamsError,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError

from multimodal_agent.agent import MultimodalAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper – build a Message with metadata
# ---------------------------------------------------------------------------

def _agent_message(
    text: str,
    context_id: str | None,
    task_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Create an agent :class:`Message` with optional *metadata*.

    This replaces the upstream ``new_agent_text_message`` helper so that
    every message can carry structured metadata describing the event type.
    """
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid.uuid4()),
        task_id=task_id,
        context_id=context_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Extracted media container
# ---------------------------------------------------------------------------

class _ExtractedParts:
    """Container for parts extracted from an A2A message."""

    def __init__(self):
        self.system_prompt: str | None = None
        self.query: str = ""
        self.image_paths: list[str] = []
        self.image_b64s: list[str] = []
        self.video_path: str | None = None
        self.video_b64: str | None = None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MultimodalAgentExecutor(AgentExecutor):
    """A2A executor that bridges the protocol to the MultimodalAgent.

    Supports:
    - Optional system prompt (first TextPart starting with ``[system]`` or
      a dedicated metadata field).
    - Multiple images (each as a FilePart with image/* MIME type).
    - One video (first FilePart with video/* MIME type).
    """

    def __init__(self, task_store: TaskStore | None = None):
        self.agent = MultimodalAgent()
        self._task_store = task_store

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        # Extract all parts from the A2A message
        parts = self._extract_parts(context)

        if not parts.query:
            raise ServerError(error=InvalidParamsError())

        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(
                query=parts.query,
                context_id=task.context_id,
                system_prompt=parts.system_prompt,
                image_paths=parts.image_paths if parts.image_paths else None,
                image_b64s=parts.image_b64s if parts.image_b64s else None,
                video_path=parts.video_path,
                video_b64=parts.video_b64,
            ):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                event_type = item.get("event_type", "token")

                content = item["content"]
                logger.info(
                    "Agent executor received event: %s, content: %s",
                    event_type,
                    content[:100] if content else "",
                )

                # ----- token (LLM streaming chunk for final response) -----
                # Tokens are streamed chunks of the final response.
                # We send them with event_type "token" for the frontend to
                # accumulate and display as a streaming message.
                if event_type == "token":
                    meta = {"event_type": "token"}
                    await updater.update_status(
                        TaskState.working,
                        _agent_message(
                            content,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                        metadata=meta,
                    )

                # ----- thought (LLM text accompanying tool calls) -----
                elif event_type == "thought":
                    meta = {"event_type": "llm_thought"}
                    await updater.update_status(
                        TaskState.working,
                        _agent_message(
                            content,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                        metadata=meta,
                    )

                # ----- tool_start → tool_call -----
                elif event_type in ("tool_start", "tool_call"):
                    tool_name = item.get("tool_name", "")
                    tool_input = item.get("tool_input", "")
                    meta: dict[str, Any] = {
                        "event_type": "tool_call",
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                    }
                    # Keep backward-compatible text prefix
                    display = f"[tool_start:{tool_name}] {content}"
                    if tool_input:
                        display += f"\n```json\n{tool_input}\n```"
                    logger.info("Sending tool_call to A2A: %s", display[:200])
                    await updater.update_status(
                        TaskState.working,
                        _agent_message(
                            display,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                        metadata=meta,
                    )

                # ----- tool_end → tool_result -----
                elif event_type == "tool_end":
                    tool_name = item.get("tool_name", "")
                    tool_output = item.get("tool_output", "")
                    meta = {
                        "event_type": "tool_result",
                        "tool_name": tool_name,
                        "tool_output": tool_output,
                    }
                    display = f"[tool_end:{tool_name}] {content}"
                    if tool_output:
                        display += f"\n```\n{tool_output}\n```"
                    logger.info("Sending tool_result to A2A: %s", display[:200])
                    await updater.update_status(
                        TaskState.working,
                        _agent_message(
                            display,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                        metadata=meta,
                    )

                # ----- require_user_input -----
                elif require_user_input:
                    meta = {"event_type": "input_required"}
                    await updater.update_status(
                        TaskState.input_required,
                        _agent_message(
                            content,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                        final=True,
                        metadata=meta,
                    )
                    break

                # ----- task complete → final_response -----
                elif is_task_complete:
                    meta = {"event_type": "final_response"}
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name="multimodal_result",
                        metadata=meta,
                    )
                    await updater.complete(
                        message=_agent_message(
                            content,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                    )
                    break

                # ----- generic status (e.g. "Processing images…") -----
                else:
                    meta = {"event_type": "status"}
                    await updater.update_status(
                        TaskState.working,
                        _agent_message(
                            content,
                            task.context_id,
                            task.id,
                            metadata=meta,
                        ),
                        metadata=meta,
                    )

        except Exception as e:
            logger.error("Error streaming agent response: %s", e, exc_info=True)
            raise ServerError(error=InternalError()) from e

    def _extract_parts(self, context: RequestContext) -> _ExtractedParts:
        """Extract system prompt, text query, images, and video from A2A message.

        Convention for system prompt:
        - A TextPart whose text starts with ``[system]`` (case-insensitive) is
          treated as the system prompt.  The ``[system]`` prefix is stripped.
        - Alternatively, if the message metadata contains a ``system_prompt``
          key, that value is used.

        Images:
        - All FileParts with ``image/*`` MIME types are collected (multiple
          images supported).

        Video:
        - Only the **first** FilePart with ``video/*`` MIME type is used.

        Returns
        -------
        _ExtractedParts
            Container with all extracted data.
        """
        result = _ExtractedParts()

        # Get the user input text as fallback
        user_input = context.get_user_input()
        if user_input:
            result.query = user_input

        # Check message metadata for system_prompt
        if context.message and hasattr(context.message, "metadata"):
            meta = context.message.metadata
            if isinstance(meta, dict) and "system_prompt" in meta:
                result.system_prompt = str(meta["system_prompt"])

        logger.info("context dict: %s", context.__dict__)

        # Scan message parts
        if context.message and hasattr(context.message, "parts"):
            text_parts: list[str] = []

            for part in context.message.parts:
                # Unwrap Part wrapper
                if hasattr(part, "root"):
                    part = part.root

                if isinstance(part, TextPart):
                    text = part.text or ""

                    # Check for [system] prefix
                    if text.strip().lower().startswith("[system]"):
                        # Extract system prompt (strip the prefix)
                        system_text = text.strip()[len("[system]"):].strip()
                        if system_text:
                            result.system_prompt = system_text
                    else:
                        text_parts.append(text)

                elif isinstance(part, FilePart):
                    self._process_file_part(part, result)

            # Combine all non-system text parts as the query
            if text_parts:
                result.query = "\n".join(text_parts)

        return result

    def _process_file_part(self, part: FilePart, result: _ExtractedParts) -> None:
        """Process a single FilePart and add it to the result container."""
        file_obj = getattr(part, "file", None)
        if not file_obj:
            return

        # Determine MIME type
        mime = getattr(file_obj, "mime_type", "") or getattr(file_obj, "mimeType", "") or ""
        is_video = mime.startswith("video/")
        is_image = mime.startswith("image/")

        # If no MIME, try to guess from URI
        if not is_video and not is_image:
            uri = getattr(file_obj, "uri", "") or ""
            if uri:
                suffix = Path(uri.split("?")[0]).suffix.lower()
                from multimodal_agent.agent import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS
                is_video = suffix in VIDEO_EXTENSIONS
                is_image = suffix in IMAGE_EXTENSIONS

        # Extract data (inline bytes or URI)
        file_b64: str | None = None
        file_path: str | None = None

        if hasattr(file_obj, "bytes") and file_obj.bytes:
            raw = file_obj.bytes
            if isinstance(raw, bytes):
                file_b64 = base64.b64encode(raw).decode()
            else:
                file_b64 = str(raw)

        elif hasattr(file_obj, "uri") and file_obj.uri:
            uri = file_obj.uri
            if uri.startswith("file://"):
                file_path = uri[7:]
            elif uri.startswith("/"):
                file_path = uri
            elif uri.startswith("data:"):
                # data URL — extract base64
                if "," in uri:
                    file_b64 = uri.split(",", 1)[1]
            # else: external URL — could be passed as text, but skip for now

        # Route to the appropriate list
        if is_video:
            # Only accept the first video
            if result.video_path is None and result.video_b64 is None:
                if file_path:
                    result.video_path = file_path
                elif file_b64:
                    result.video_b64 = file_b64
                else:
                    logger.warning("Video FilePart has no extractable data")
            else:
                logger.warning("Multiple videos provided; only the first is used")

        elif is_image:
            if file_path:
                result.image_paths.append(file_path)
            elif file_b64:
                result.image_b64s.append(file_b64)
            else:
                logger.warning("Image FilePart has no extractable data")

        else:
            # Unknown type — try as image
            logger.info("Unknown MIME '%s' — treating as image", mime)
            if file_path:
                result.image_paths.append(file_path)
            elif file_b64:
                result.image_b64s.append(file_b64)

    def _validate_request(self, context: RequestContext) -> bool:
        # Return False = no error (valid request)
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
