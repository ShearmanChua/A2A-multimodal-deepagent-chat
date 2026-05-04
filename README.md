# A2A Multimodal DeepAgent Chat

A full-stack multimodal AI agent system that enables users to interact with AI agents through text, images, and videos using the [A2A (Agent-to-Agent) protocol](https://github.com/google/A2A). The system features a React-based chat frontend, a Python-based multimodal agent powered by LangGraph DeepAgent, and an MCP (Model Context Protocol) tool server for specialized capabilities like target detection and classification.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│                     (React + Vite + Tailwind CSS)                           │
│                         Port 3002 (via Express)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ A2A Protocol (JSON-RPC)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multimodal DeepAgent                                  │
│                    (LangGraph + LangChain + A2A SDK)                        │
│                              Port 10010                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. Extract text + media from A2A message                           │    │
│  │  2. Upload media to MinIO via boto3                                 │    │
│  │  3. Generate pre-signed URLs                                        │    │
│  │  4. Build LLM message with image_url blocks + text URLs             │    │
│  │  5. Run LangGraph DeepAgent with MCP tools                          │    │
│  │  6. Stream response via A2A protocol                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                           │
          │ boto3 (S3 API)                            │ HTTP (FastMCP)
          ▼                                           ▼
┌──────────────────────┐                 ┌──────────────────────────────────┐
│       MinIO          │                 │        MCP Tool Server           │
│   (Object Storage)   │                 │         (FastMCP)                │
│   Ports 9000/9001    │                 │          Port 8000               │
│                      │                 │  ┌────────────────────────────┐  │
│  • Image storage     │                 │  │ • DuckDuckGo Search        │  │
│  • Video storage     │                 │  │ • Target Detection (YOLO)  │  │
│  • Pre-signed URLs   │                 │  │ • Target Classification    │  │
└──────────────────────┘                 │  │ • VLM Detection/Class.     │  │
                                         │  │ • MinIO Object Access      │  │
                                         │  └────────────────────────────┘  │
                                         └──────────────────────────────────┘
                                                        │
                                                        ▼
                                         ┌──────────────────────────────────┐
                                         │         Phoenix (Arize)          │
                                         │      (Observability/Tracing)     │
                                         │       Ports 6006 / 4317          │
                                         └──────────────────────────────────┘
```

## 📁 Project Structure

```
A2A-multimodal-deepagent-chat/
├── build/                          # Docker Compose orchestration
│   ├── docker-compose.yml          # Main compose file for all services
│   ├── .env.example                # Environment variable template
│   └── .env                        # Local environment configuration
│
├── multimodal_chat_frontend/       # React-based chat interface
│   ├── client/                     # Vite + React + Tailwind frontend
│   │   ├── src/
│   │   │   ├── App.jsx             # Main application component
│   │   │   ├── api.js              # API client for backend
│   │   │   └── components/         # React components
│   │   │       ├── AgentRegistry.jsx   # Agent management UI
│   │   │       ├── AgentSelector.jsx   # Agent selection dropdown
│   │   │       ├── ChatView.jsx        # Chat interface
│   │   │       └── Header.jsx          # Application header
│   │   └── package.json
│   ├── server/                     # Express.js backend
│   │   ├── index.js                # Server entry point
│   │   ├── a2aClient.js            # A2A protocol client
│   │   └── package.json
│   └── build/                      # Docker build files
│
├── multimodal_deepagent/           # Python A2A agent
│   ├── src/
│   │   ├── main.py                 # A2A server entry point
│   │   ├── test_client.py          # Test client for the agent
│   │   └── multimodal_agent/
│   │       ├── agent.py            # LangGraph DeepAgent implementation
│   │       ├── agent_executor.py   # A2A AgentExecutor bridge
│   │       └── minio_uploader.py   # MinIO upload + pre-signed URLs
│   ├── pyproject.toml              # Python dependencies (uv)
│   └── build/Dockerfile
│
└── mcp_tools/                      # MCP tool server
    ├── src/
    │   ├── server.py               # FastMCP server entry point
    │   └── tools/                  # Tool implementations
    │       ├── duckduckgo.py       # Web search
    │       ├── target_detection.py # YOLO-based detection
    │       ├── target_classification.py
    │       ├── vlm_target_detection.py    # VLM-based detection
    │       ├── vlm_target_classification.py
    │       ├── minio_store.py      # MinIO object access
    │       └── ...
    └── build/Dockerfile
```

## 🔄 Workflow

### 1. User Interaction Flow

```
User → Chat Frontend → Express Server → A2A Protocol → DeepAgent → LLM + Tools → Response
```

1. **User sends a message** (text, image, or video) through the React chat interface
2. **Express server** receives the request and forwards it to the registered A2A agent
3. **A2A protocol** handles the JSON-RPC communication with streaming support
4. **DeepAgent** processes the message:
   - Extracts media from the A2A message
   - Uploads media to MinIO and generates pre-signed URLs
   - Builds a multimodal LLM message with image_url blocks
   - Executes the LangGraph agent with MCP tools
5. **Response streams back** through SSE (Server-Sent Events) to the frontend

### 2. Media Processing Flow

```
Image/Video → Base64 Encoding → MinIO Upload → Pre-signed URL → LLM + MCP Tools
```

- **Images**: Uploaded to MinIO, pre-signed URLs passed to both the vision model and MCP tools
- **Videos**: Frames are sampled (configurable, default 8 frames), each frame uploaded separately
- **Two modes**: 
  - `base64` mode: Images sent as data URLs to LLM, MinIO URLs for MCP tools
  - `minio` mode: MinIO URLs used for both LLM and tools (requires accessible endpoint)

### 3. Tool Execution Flow

```
Agent → MCP Client → MCP Server → Tool Execution → Result → Agent
```

Available MCP tools:
- **duckduckgo_search**: Web search via DuckDuckGo
- **search_images**: Image search via DuckDuckGo
- **target_detection**: YOLO-based object detection for vehicles, vessels, aircraft, etc.
- **target_classification**: Classify detected targets with threat assessment
- **vlm_target_detection**: High-quality VLM-based detection (slower but more accurate)
- **vlm_target_classification**: Detailed VLM-based classification with NATO designations
- **list_minio_objects**: Browse MinIO bucket contents
- **get_minio_object**: Retrieve objects from MinIO

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) `uv` for local Python development
- (Optional) Node.js 18+ for local frontend development

### 1. Clone and Configure

```bash
git clone <repository-url>
cd A2A-multimodal-deepagent-chat

# Copy environment template
cp build/.env.example build/.env

# Edit .env with your configuration
vim build/.env
```

### 2. Configure Environment Variables

Edit `build/.env` with your settings:

```bash
# Required: LLM Configuration
MODEL_ENDPOINT=https://api.openai.com/v1  # Or your LLM endpoint
MODEL_NAME=gpt-4o
MODEL_API_KEY=sk-your-api-key

# Optional: VLM for high-quality detection/classification
VLM_ENDPOINT=https://api.openai.com/v1
VLM_MODEL=gpt-4o
VLM_API_KEY=sk-your-api-key

# MinIO (defaults work for local development)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=data

# Image mode: "base64" (default, works with OpenAI) or "minio" (for local models)
IMAGE_MODE=base64
```

### 3. Start All Services

```bash
cd build
docker compose up -d
```

This starts:
- **multimodal-chat**: Frontend + Express server on port `3002`
- **multimodal-deepagent**: A2A agent on port `10010`
- **mcp-server**: MCP tool server on port `8000`
- **minio**: Object storage on ports `9000` (API) and `9001` (console)
- **phoenix**: Observability UI on port `6006`

### 4. Access the Application

1. Open the chat interface: http://localhost:3002
2. Register the DeepAgent:
   - Click "Add Agent" in the Agent Registry
   - Enter URL: `http://multimodal-deepagent:10010` (or `http://localhost:10010` if accessing from host)
   - The agent card will be fetched automatically
3. Select the agent and start chatting!

### 5. Optional: Access Other Services

- **MinIO Console**: http://localhost:9001 (login: minioadmin/minioadmin)
- **Phoenix Tracing**: http://localhost:6006

## 🛠️ Development

### Local Development (Frontend)

```bash
# Terminal 1: Start the server
cd multimodal_chat_frontend/server
npm install
npm run dev

# Terminal 2: Start the client
cd multimodal_chat_frontend/client
npm install
npm run dev

# Access at http://localhost:5173
```

### Local Development (Agent)

```bash
cd multimodal_deepagent

# Install dependencies with uv
uv sync

# Set environment variables
export MODEL_NAME=gpt-4o
export MODEL_API_KEY=sk-...
export MINIO_ENDPOINT=localhost:9000
export MCP_SERVER_URL=http://localhost:8000

# Start the agent
uv run python src/main.py --host 0.0.0.0 --port 10010
```

### Testing the Agent

```bash
cd multimodal_deepagent

# Text-only query
uv run python src/test_client.py --url http://localhost:10010

# With an image
uv run python src/test_client.py --url http://localhost:10010 --image /path/to/image.jpg

# Custom query
uv run python src/test_client.py --url http://localhost:10010 \
  --query "Detect targets in this image" \
  --image /path/to/image.jpg
```

## 📡 A2A Protocol

This project implements the [A2A (Agent-to-Agent) protocol](https://github.com/google/A2A) for agent communication:

### Agent Discovery

```bash
# Fetch agent card
curl http://localhost:10010/.well-known/agent.json
```

### Message Send (JSON-RPC)

```json
{
  "jsonrpc": "2.0",
  "id": "unique-id",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        { "kind": "text", "text": "Describe this image" },
        { "kind": "file", "file": { "bytes": "<base64>", "mimeType": "image/jpeg" } }
      ],
      "messageId": "msg-id",
      "contextId": "conversation-id"
    }
  }
}
```

### Streaming (SSE)

Use `method: "message/stream"` for streaming responses via Server-Sent Events.

## 🔧 Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o` | LLM model name |
| `MODEL_ENDPOINT` | OpenAI default | LLM API base URL |
| `MODEL_API_KEY` | `EMPTY` | LLM API key |
| `MCP_SERVER_URL` | `http://research-mcp-server-1:8000` | MCP tool server URL |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO |
| `MINIO_BUCKET` | `data` | Default bucket name |
| `MINIO_EXTERNAL_ENDPOINT` | Same as `MINIO_ENDPOINT` | External endpoint for pre-signed URLs |
| `MINIO_PRESIGN_EXPIRY` | `3600` | Pre-signed URL expiry (seconds) |
| `MAX_VIDEO_FRAMES` | `8` | Number of frames to sample from videos |
| `IMAGE_MODE` | `base64` | Image delivery mode: `base64` or `minio` |
| `VLM_ENDPOINT` | Falls back to `MODEL_ENDPOINT` | VLM endpoint for high-quality detection |
| `VLM_MODEL` | Falls back to `MODEL_NAME` | VLM model name |
| `VLM_API_KEY` | Falls back to `MODEL_API_KEY` | VLM API key |

### Docker Compose Services

| Service | Port(s) | Description |
|---------|---------|-------------|
| `multimodal-chat` | 3002 | Frontend + Express server |
| `multimodal-deepagent` | 10010 | A2A multimodal agent |
| `mcp-server` | 8000 | MCP tool server |
| `minio` | 9000, 9001 | Object storage (API, Console) |
| `phoenix` | 6006, 4317 | Observability (UI, OTLP) |

## 📚 Key Technologies

- **Frontend**: React 19, Vite, Tailwind CSS, Lucide Icons
- **Backend**: Express.js, Node.js
- **Agent**: Python 3.12+, LangGraph, LangChain, DeepAgents, A2A SDK
- **Tools**: FastMCP, YOLOv8, OpenCV
- **Storage**: MinIO (S3-compatible)
- **Observability**: Arize Phoenix, OpenTelemetry
- **Protocol**: A2A (Agent-to-Agent), MCP (Model Context Protocol)

## 📄 License

MIT
