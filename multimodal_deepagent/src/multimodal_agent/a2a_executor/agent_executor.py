"""
A2A AgentExecutor for the Multimodal DeepAgent.

Handles incoming A2A requests, extracts text, system prompt, images,
or ONE video from message parts, delegates to the MultimodalAgent, and
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
    FileWithBytes,
    FileWithUri,
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

from multimodal_agent.agent.agent import MultimodalAgent

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
        self.image_urls: list[str] = []
        self.video_path: str | None = None
        self.video_b64: str | None = None
        self.video_urls: list[str] = []
        # (virtual_filename, markdown_text) for each uploaded document
        self.documents: list[tuple[str, str]] = []
        # (orig_name, uri, mime) for URI-based documents that need async download
        self.document_uris: list[tuple[str, str, str]] = []


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MultimodalAgentExecutor(AgentExecutor):
    """A2A executor that bridges the protocol to the MultimodalAgent.

    Supports:
    - Optional system prompt via ``MessageSendParams.metadata["system_prompt"]``.
    - Multiple images (each as a FilePart with image/* MIME type, or http/https URI).
    - One inline video (first FilePart with video/* MIME type, as bytes or file path).
    - Multiple video URLs (http/https URIs with video/* MIME type — passed directly
      to the model without frame sampling).
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

        # Download and convert any URI-based documents (presigned URLs from object store)
        if parts.document_uris:
            from multimodal_agent.agent.doc_converter import convert_to_markdown, virtual_filename
            import httpx
            async with httpx.AsyncClient() as http_client:
                for orig_name, uri, mime in parts.document_uris:
                    try:
                        resp = await http_client.get(uri, timeout=60.0, follow_redirects=True)
                        resp.raise_for_status()
                        md_text = convert_to_markdown(resp.content, mime, orig_name)
                        vname = virtual_filename(orig_name)
                        parts.documents.append((vname, md_text))
                        logger.info(
                            "Downloaded and converted '%s' → /uploads/%s (%d chars)",
                            orig_name, vname, len(md_text),
                        )
                    except Exception as exc:
                        logger.error("Failed to download document '%s' from URI: %s", orig_name, exc)

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
                image_urls=parts.image_urls if parts.image_urls else None,
                video_path=parts.video_path,
                video_b64=parts.video_b64,
                video_urls=parts.video_urls if parts.video_urls else None,
                documents=parts.documents if parts.documents else None,
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
        """Extract system prompt, text query, images, video, and documents from A2A message.

        System prompt:
        - Read from ``MessageSendParams.metadata["system_prompt"]``.

        Documents:
        - FileParts whose MIME type is in DOCUMENT_MIMES are collected in order.
        - Their original filenames come from ``metadata["document_names"]``
          (a parallel list set by the Express server).
        - Each is converted to markdown by ``doc_converter.convert_to_markdown``.

        Images:
        - All FileParts with ``image/*`` MIME types are collected.

        Video:
        - Only the first FilePart with ``video/*`` MIME type is used.
        """
        from multimodal_agent.agent.doc_converter import convert_to_markdown, is_document, virtual_filename

        result = _ExtractedParts()

        user_input = context.get_user_input()
        if user_input:
            result.query = user_input

        params_meta = context.metadata  # dict[str, Any], never None
        if "system_prompt" in params_meta:
            result.system_prompt = str(params_meta["system_prompt"])

        # Original filenames for document parts, sent by Express as metadata
        doc_names: list[str] = list(params_meta.get("document_names", []))

        logger.info("context dict: %s", context.__dict__)

        if context.message and hasattr(context.message, "parts"):
            text_parts: list[str] = []
            # Collect document FileParts in order so we can zip with doc_names
            doc_file_parts: list[FilePart] = []

            for part in context.message.parts:
                if hasattr(part, "root"):
                    part = part.root

                if isinstance(part, TextPart):
                    text_parts.append(part.text or "")
                elif isinstance(part, FilePart):
                    mime = (part.file.mime_type or "") if hasattr(part.file, "mime_type") else ""
                    if is_document(mime):
                        doc_file_parts.append(part)
                    else:
                        self._process_file_part(part, result)

            if text_parts:
                result.query = "\n".join(text_parts)

            # Convert document parts → markdown (URI-based deferred to execute())
            for file_part, orig_name in zip(doc_file_parts, doc_names):
                if isinstance(file_part.file, FileWithBytes):
                    try:
                        raw = base64.b64decode(file_part.file.bytes)
                        mime = file_part.file.mime_type or ""
                        md_text = convert_to_markdown(raw, mime, orig_name)
                        vname = virtual_filename(orig_name)
                        result.documents.append((vname, md_text))
                        logger.info("Converted '%s' → /uploads/%s (%d chars)", orig_name, vname, len(md_text))
                    except Exception as exc:
                        logger.error("Failed to convert document '%s': %s", orig_name, exc)
                elif isinstance(file_part.file, FileWithUri):
                    mime = file_part.file.mime_type or ""
                    result.document_uris.append((orig_name, file_part.file.uri, mime))
                    logger.info("Queued document '%s' for async download", orig_name)
                else:
                    logger.warning("Document FilePart has unknown file type, skipping: %s", orig_name)

        return result

    def _process_file_part(self, part: FilePart, result: _ExtractedParts) -> None:
        """Process a single FilePart and add it to the result container."""
        file = part.file
        mime = file.mime_type or ""
        is_image = mime.startswith("image/")
        is_video = mime.startswith("video/")

        if isinstance(file, FileWithBytes):
            # file.bytes is already a base64-encoded string per the A2A spec
            self._route_media(file.bytes, None, is_image, is_video, mime, result)

        elif isinstance(file, FileWithUri):
            uri = file.uri

            # Guess type from extension when MIME is absent
            if not is_image and not is_video:
                from multimodal_agent.configs.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
                suffix = Path(uri.split("?")[0]).suffix.lower()
                is_image = suffix in IMAGE_EXTENSIONS
                is_video = suffix in VIDEO_EXTENSIONS

            if uri.startswith(("http://", "https://")):
                if is_video:
                    result.video_urls.append(uri)
                else:
                    result.image_urls.append(uri)
                return

            if uri.startswith("data:") and "," in uri:
                self._route_media(uri.split(",", 1)[1], None, is_image, is_video, mime, result)
                return

            if uri.startswith("file://"):
                file_path = uri[len("file://"):]
            elif uri.startswith("/"):
                file_path = uri
            else:
                logger.warning("Unrecognised URI scheme in FilePart: %s", uri[:80])
                return

            self._route_media(None, file_path, is_image, is_video, mime, result)

    def _route_media(
        self,
        file_b64: str | None,
        file_path: str | None,
        is_image: bool,
        is_video: bool,
        mime: str,
        result: _ExtractedParts,
    ) -> None:
        """Route an extracted file to the correct bucket on *result*."""
        if is_video:
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
