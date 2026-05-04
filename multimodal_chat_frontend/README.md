# Multimodal Chat Frontend

A React-based chat interface for communicating with A2A (Agent-to-Agent) compatible agents.

## Features

- **Agent Registry**: Register and manage multiple A2A agents
- **Agent Selection**: Choose from registered agents to chat with
- **Multimodal Input**: Send text, images, and videos to agents
- **Agent Info Display**: View agent skills, capabilities, and supported modalities
- **Session Management**: Maintain conversation context across messages

## Architecture

```
multimodal_chat_frontend/
├── client/                 # React frontend (Vite + Tailwind)
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── api.js          # API client
│   │   └── App.jsx         # Main app component
│   └── package.json
├── server/                 # Express backend
│   ├── index.js            # Server entry point
│   ├── a2aClient.js        # A2A protocol client
│   └── package.json
└── build/                  # Docker build files
    ├── Dockerfile          # Combined client + server
    ├── Dockerfile.client   # Client only (nginx)
    ├── Dockerfile.server   # Server only
    └── nginx.conf          # Nginx configuration
```

## Quick Start

### Development

1. Install dependencies:

```bash
# Client
cd client && npm install

# Server
cd server && npm install
```

2. Start the server:

```bash
cd server && npm run dev
```

3. Start the client:

```bash
cd client && npm run dev
```

4. Open http://localhost:5173

### Docker

Build and run with Docker Compose:

```bash
docker compose -f build/docker-compose.chat.yml up -d
```

## API Endpoints

### Agent Registry

- `GET /api/agents` - List registered agents
- `POST /api/agents` - Register a new agent
- `DELETE /api/agents/:id` - Remove an agent
- `POST /api/agents/:id/refresh` - Refresh agent status
- `POST /api/agents/fetch-card` - Preview agent card before registering

### Chat

- `POST /api/chat` - Send a text message
- `POST /api/chat/upload` - Upload media and send to agent

### Sessions

- `GET /api/sessions` - List sessions
- `POST /api/sessions` - Create a new session
- `GET /api/sessions/:id` - Get session details
- `POST /api/sessions/:id/chat` - Send message in session
- `POST /api/sessions/:id/upload` - Upload media in session

## A2A Protocol

This frontend communicates with agents using the A2A (Agent-to-Agent) protocol:

1. **Agent Card**: Fetched from `/.well-known/agent.json` to discover agent capabilities
2. **Message Send**: JSON-RPC `message/send` method for sending messages
3. **Multimodal Support**: Images and videos are sent as base64-encoded file parts

### Supported Input Modes

The frontend detects supported input modes from the agent card:
- `text`, `text/plain` - Text messages
- `image/png`, `image/jpeg` - Image uploads
- `video/mp4` - Video uploads

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | `3002` | Server port |
| `NODE_ENV` | `development` | Environment mode |

## License

MIT
