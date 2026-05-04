"""
Multimodal DeepAgent A2A Server — Entry Point.

Starts an A2A-protocol server that accepts text, images, and videos.
Media is uploaded to MinIO via boto3 and pre-signed URLs are passed to
the LLM agent and MCP tools.

Environment variables:
    MODEL_NAME, MODEL_ENDPOINT, MODEL_API_KEY  — LLM configuration
    MCP_SERVER_URL      — MCP tool server (default http://research-mcp-server-1:8000)
    MINIO_ENDPOINT      — MinIO S3 endpoint (default minio:9000)
    MINIO_ACCESS_KEY    — MinIO access key (default minioadmin)
    MINIO_SECRET_KEY    — MinIO secret key (default minioadmin)
    MINIO_SECURE        — Use HTTPS for MinIO (default false)
    MINIO_BUCKET        — Default bucket (default data)
    MINIO_EXTERNAL_ENDPOINT — External MinIO endpoint for pre-signed URLs
    MINIO_PRESIGN_EXPIRY    — Pre-signed URL expiry in seconds (default 3600)
"""

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
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from multimodal_agent.agent import MultimodalAgent
from multimodal_agent.agent_executor import MultimodalAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingConfigError(Exception):
    """Raised when required configuration is missing."""


@click.command()
@click.option("--host", "host", default="0.0.0.0")
@click.option("--port", "port", default=10010)
def main(host, port):
    """Start the Multimodal DeepAgent A2A server."""
    try:
        # Validate minimum configuration
        if not os.getenv("MODEL_NAME") and not os.getenv("TOOL_LLM_NAME"):
            logger.warning(
                "Neither MODEL_NAME nor TOOL_LLM_NAME is set. "
                "Falling back to 'gpt-4o'."
            )

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        skills = [
            AgentSkill(
                id="multimodal_analysis",
                name="Multimodal Analysis",
                description=(
                    "Analyse images and videos using vision models and MCP tools. "
                    "Supports object detection, target classification, and general "
                    "visual question answering. Media is uploaded to MinIO and "
                    "pre-signed URLs are used for tool integration."
                ),
                tags=[
                    "image analysis",
                    "video analysis",
                    "object detection",
                    "target classification",
                    "multimodal",
                ],
                examples=[
                    "Describe what you see in this image.",
                    "Detect all targets in the uploaded image.",
                    "Classify the vehicle in the bounding box.",
                    "What objects are present in this video?",
                ],
            ),
            AgentSkill(
                id="minio_media_management",
                name="MinIO Media Management",
                description=(
                    "Upload images and videos to MinIO object storage and "
                    "generate pre-signed URLs for sharing and tool access."
                ),
                tags=["minio", "upload", "storage", "pre-signed URL"],
                examples=[
                    "Upload this image and give me the URL.",
                    "List objects in the MinIO bucket.",
                ],
            ),
        ]

        agent_card = AgentCard(
            name="Multimodal DeepAgent",
            description=(
                "A multimodal research agent that processes images and videos. "
                "Media is uploaded to MinIO storage and pre-signed URLs are "
                "generated for the agent and MCP tools to use."
            ),
            url=os.environ.get("A2A_AGENT_URL", "http://localhost:10010"),
            version="1.0.0",
            default_input_modes=MultimodalAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store,
        )
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=MultimodalAgentExecutor(task_store=task_store),
            task_store=task_store,
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        logger.info(
            "Starting Multimodal DeepAgent A2A server on %s:%d", host, port
        )
        logger.info(
            "MinIO endpoint: %s | MCP server: %s",
            os.environ.get("MINIO_ENDPOINT", "minio:9000"),
            os.environ.get("MCP_SERVER_URL", "http://research-mcp-server-1:8000"),
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingConfigError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Server startup failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
