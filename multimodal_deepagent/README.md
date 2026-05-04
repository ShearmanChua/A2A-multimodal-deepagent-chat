# Multimodal DeepAgent (A2A)

A multimodal A2A (Agent-to-Agent) agent that processes images and videos from
users using MCP tools. Instead of passing base64-encoded media directly to the
agent, media is uploaded to **MinIO** object storage via **boto3** and
**pre-signed URLs** are generated and passed to the agent.

## Architecture

```
User (A2A Client)
  │
  ├─ text query + optional image/video (base64 or file)
  │
  ▼
┌─────────────────────────────────────────┐
│  Multimodal DeepAgent (A2A Server)      │
│                                         │
│  1. Extract text + media from A2A msg   │
│  2. Upload media to MinIO via boto3     │
│  3. Generate pre-signed URL             │
│  4. Build LLM message with:             │
│     - image_url blocks (vision model)   │
│     - text with URLs (for MCP tools)    │
│  5. Run LangGraph react agent           │
│     - Calls MCP tools with URLs         │
│  6. Return result via A2A protocol      │
└─────────────────────────────────────────┘
  │                          │
  ▼                          ▼
┌──────────┐          ┌──────────────┐
│  MinIO   │          │  MCP Server  │
│ (S3 API) │          │  (tools)     │
└──────────┘          └──────────────┘
```

## Key Differences from `agents/src/gradio_deepagent_app.py`

| Feature | Original (Gradio) | This Agent (A2A) |
|---------|-------------------|------------------|
| Protocol | Gradio web UI | A2A protocol (HTTP/JSON-RPC) |
| Media handling | Base64 passed to MCP `upload_image` tool | Uploaded to MinIO via boto3, pre-signed URL generated |
| Image delivery to agent | Base64 data-URL in `image_url` block | Pre-signed MinIO URL in `image_url` block |
| Image delivery to tools | Base64 via MCP `upload_image` → `image_id` | Pre-signed URL passed as text to tools |
| Video handling | Frames sampled → base64 data-URLs | Frames sampled → uploaded to MinIO → pre-signed URLs |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o` | LLM model name |
| `MODEL_ENDPOINT` | *(OpenAI default)* | LLM base URL |
| `MODEL_API_KEY` | `EMPTY` | LLM API key |
| `MCP_SERVER_URL` | `http://research-mcp-server-1:8000` | MCP tool server URL |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO |
| `MINIO_BUCKET` | `data` | Default bucket name |
| `MINIO_EXTERNAL_ENDPOINT` | *(same as MINIO_ENDPOINT)* | External endpoint for pre-signed URLs |
| `MINIO_PRESIGN_EXPIRY` | `3600` | Pre-signed URL expiry (seconds) |
| `MAX_VIDEO_FRAMES` | `8` | Number of frames to sample from videos |

## Running Locally

```bash
cd multimodal_deepagent

# Install dependencies
uv sync

# Set environment variables
export MODEL_NAME=gpt-4o
export MODEL_API_KEY=sk-...
export MINIO_ENDPOINT=localhost:9000

# Start the server
uv run python src/main.py --host 0.0.0.0 --port 10010
```

## Testing

```bash
# Text-only query
uv run python src/test_client.py --url http://localhost:10010

# With an image
uv run python src/test_client.py --url http://localhost:10010 --image /path/to/image.jpg

# Custom query
uv run python src/test_client.py --url http://localhost:10010 --query "Detect targets in this image" --image /path/to/image.jpg
```

## Docker

```bash
docker build -f build/Dockerfile -t multimodal-deepagent .
docker run -p 10010:10010 \
  -e MODEL_NAME=gpt-4o \
  -e MODEL_API_KEY=sk-... \
  -e MINIO_ENDPOINT=minio:9000 \
  multimodal-deepagent
```

## File Structure

```
multimodal_deepagent/
├── build/
│   └── Dockerfile
├── pyproject.toml
├── README.md
└── src/
    ├── main.py                          # A2A server entry point
    ├── test_client.py                   # Test client
    └── multimodal_agent/
        ├── __init__.py
        ├── agent.py                     # Core LangGraph agent with MinIO + MCP
        ├── agent_executor.py            # A2A AgentExecutor bridge
        └── minio_uploader.py            # boto3 MinIO upload + pre-signed URLs
```
