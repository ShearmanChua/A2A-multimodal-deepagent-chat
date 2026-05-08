"""Tool helpers — error wrapping and response schema."""

from __future__ import annotations

from functools import wraps
from typing import Literal

from langchain_core.tools import ToolException
from pydantic import BaseModel


class ResponseFormat(BaseModel):
    """Structured response emitted by the fallback react-agent."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


def wrap_tools_with_error_handling(tools: list) -> list:
    """Wrap every tool so unhandled exceptions become friendly LLM error strings.

    MCP tools (StructuredTool) are async-only; for those only ``_arun`` is
    wrapped.  Sync-capable tools have both ``_run`` and ``_arun`` wrapped.
    """
    def _error_formatter(exc: ToolException) -> str:
        return (
            f"⚠️ Tool error: {exc}\n"
            "You may retry with corrected parameters, try an alternative tool, "
            "or skip this step."
        )

    for tool in tools:
        tool.handle_tool_error = _error_formatter

        is_async_only = (
            hasattr(tool, "coroutine") and tool.coroutine is not None
        )

        if not is_async_only and hasattr(tool, "_run"):
            original_run = tool._run

            def _make_safe_run(orig):
                @wraps(orig)
                def _safe_run(*args, **kwargs):
                    try:
                        return orig(*args, **kwargs)
                    except (ToolException, NotImplementedError):
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
