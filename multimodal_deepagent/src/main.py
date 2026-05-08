"""Multimodal DeepAgent A2A Server — entry point."""

import logging
import os
import sys

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from multimodal_agent.agent.agent import MultimodalAgent
from multimodal_agent.a2a_executor.agent_executor import MultimodalAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=10010)
def main(host: str, port: int) -> None:
    """Start the Multimodal DeepAgent A2A server."""
    try:
        if not os.getenv("MODEL_NAME") and not os.getenv("TOOL_LLM_NAME"):
            logger.warning("Neither MODEL_NAME nor TOOL_LLM_NAME is set — falling back to 'gpt-4o'.")

        skills = [
            AgentSkill(
                id="multimodal_rag",
                name="Multimodal RAG",
                description=(
                    "Answer questions by searching Weaviate vector collections with hybrid "
                    "search (BM25 + vector similarity). Accepts text queries and images; "
                    "images are uploaded to SeaweedFS and pre-signed URLs are shared with "
                    "MCP tools for multimodal retrieval."
                ),
                tags=["rag", "weaviate", "vector search", "multimodal", "image"],
                examples=[
                    "What documents mention transformer architectures?",
                    "Find images similar to the one I uploaded.",
                    "Search the knowledge base for information about this diagram.",
                ],
            ),
            AgentSkill(
                id="media_management",
                name="Media Management",
                description=(
                    "Upload images and videos to SeaweedFS object storage and browse "
                    "objects stored in MinIO. Generates pre-signed URLs for sharing "
                    "and MCP tool access."
                ),
                tags=["seaweedfs", "minio", "storage", "upload", "pre-signed URL"],
                examples=[
                    "Upload this image and give me the URL.",
                    "List objects in the MinIO bucket.",
                ],
            ),
        ]

        agent_card = AgentCard(
            name="Multimodal DeepAgent",
            description=(
                "A multimodal RAG agent that processes text and images. "
                "Media is uploaded to SeaweedFS and pre-signed URLs are used "
                "for both the vision model and MCP tools."
            ),
            url=os.environ.get("A2A_AGENT_URL", "http://localhost:10010"),
            version="1.0.0",
            default_input_modes=MultimodalAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=["text", "text/plain"],
            capabilities=AgentCapabilities(streaming=True, push_notifications=True),
            skills=skills,
        )

        task_store = InMemoryTaskStore()
        push_config_store = InMemoryPushNotificationConfigStore()
        request_handler = DefaultRequestHandler(
            agent_executor=MultimodalAgentExecutor(task_store=task_store),
            task_store=task_store,
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(
                httpx_client=httpx.AsyncClient(),
                config_store=push_config_store,
            ),
        )

        logger.info("Starting Multimodal DeepAgent on %s:%d", host, port)
        logger.info(
            "SeaweedFS: %s | MCP: %s",
            os.environ.get("SEAWEEDFS_ENDPOINT", "not configured"),
            os.environ.get("MCP_SERVER_URL", "http://research-mcp-server-1:8000"),
        )

        uvicorn.run(A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler).build(), host=host, port=port)

    except Exception as e:
        logger.error("Server startup failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
