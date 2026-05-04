/**
 * Backend server for the A2A Multi-Agent Chat Interface.
 *
 * Provides:
 * - Agent registry management (list, register, remove, refresh agents)
 * - A2A protocol communication with registered agents
 * - Media upload and forwarding to agents
 * - Session management for conversations
 */

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { v4: uuidv4 } = require("uuid");

const { A2AClient } = require("./a2aClient");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env.SERVER_PORT || "3002", 10);
const UPLOAD_DIR = path.join(__dirname, "tmp", "uploads");
const AGENTS_FILE = path.join(__dirname, "data", "agents.json");

// Ensure directories exist
[UPLOAD_DIR, path.dirname(AGENTS_FILE)].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// ---------------------------------------------------------------------------
// Express setup
// ---------------------------------------------------------------------------

const app = express();

app.use(cors());
app.use(express.json({ limit: "100mb" }));
app.use(express.urlencoded({ extended: true, limit: "100mb" }));

// Global request logger - log ALL incoming requests
app.use((req, res, next) => {
  console.log(`>>> [REQUEST] ${req.method} ${req.path}`);
  next();
});

// Multer for file uploads
const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
});

// Serve static files from React build in production
if (process.env.NODE_ENV === "production") {
  const clientDistPath = path.join(__dirname, "client", "dist");
  if (fs.existsSync(clientDistPath)) {
    app.use(express.static(clientDistPath));
  } else {
    // Fallback for development structure
    const altPath = path.join(__dirname, "..", "client", "dist");
    if (fs.existsSync(altPath)) {
      app.use(express.static(altPath));
    }
  }
}

// ---------------------------------------------------------------------------
// In-memory state
// ---------------------------------------------------------------------------

// Agent registry: id -> { id, name, url, description, status, agentCard, ... }
let agents = new Map();

// Sessions: sessionId -> { sessionId, agentId, contextId, history, systemPrompt, ... }
const sessions = new Map();

// Load agents from file on startup
function loadAgents() {
  try {
    if (fs.existsSync(AGENTS_FILE)) {
      const data = JSON.parse(fs.readFileSync(AGENTS_FILE, "utf-8"));
      agents = new Map(data.map((a) => [a.id, a]));
      console.log(`Loaded ${agents.size} agents from file`);
    }
  } catch (err) {
    console.error("Failed to load agents:", err.message);
  }
}

// Save agents to file
function saveAgents() {
  try {
    const data = Array.from(agents.values());
    fs.writeFileSync(AGENTS_FILE, JSON.stringify(data, null, 2));
  } catch (err) {
    console.error("Failed to save agents:", err.message);
  }
}

// Initialize
loadAgents();

// ---------------------------------------------------------------------------
// REST API Routes
// ---------------------------------------------------------------------------

/**
 * GET /api/health - Health check
 */
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// =============================================================================
// Agent Registry API
// =============================================================================

/**
 * GET /api/agents - List all registered agents
 */
app.get("/api/agents", (req, res) => {
  const agentList = Array.from(agents.values()).map((a) => ({
    id: a.id,
    name: a.name,
    url: a.url,
    description: a.description,
    status: a.status,
    version: a.agentCard?.version,
    capabilities: a.agentCard?.capabilities,
    skills: a.agentCard?.skills,
    defaultInputModes: a.agentCard?.default_input_modes || a.agentCard?.defaultInputModes,
    defaultOutputModes: a.agentCard?.default_output_modes || a.agentCard?.defaultOutputModes,
    registeredAt: a.registeredAt,
    lastChecked: a.lastChecked,
  }));
  res.json({ agents: agentList });
});

/**
 * POST /api/agents - Register a new agent
 */
app.post("/api/agents", async (req, res) => {
  const { url, name, description } = req.body;

  if (!url) {
    return res.status(400).json({ error: "Agent URL is required" });
  }

  // Check if already registered
  const existing = Array.from(agents.values()).find((a) => a.url === url);
  if (existing) {
    return res.status(409).json({ error: "Agent already registered", agentId: existing.id });
  }

  try {
    // Fetch agent card
    const client = new A2AClient(url);
    const agentCard = await client.getAgentCard();

    const agent = {
      id: uuidv4(),
      url: url.replace(/\/$/, ""), // Remove trailing slash
      name: name || agentCard.name || "Unknown Agent",
      description: description || agentCard.description || "",
      status: "online",
      agentCard,
      registeredAt: new Date().toISOString(),
      lastChecked: new Date().toISOString(),
    };

    agents.set(agent.id, agent);
    saveAgents();

    res.json({
      agent: {
        id: agent.id,
        name: agent.name,
        url: agent.url,
        description: agent.description,
        status: agent.status,
        version: agentCard.version,
        capabilities: agentCard.capabilities,
        skills: agentCard.skills,
        defaultInputModes: agentCard.default_input_modes || agentCard.defaultInputModes,
        defaultOutputModes: agentCard.default_output_modes || agentCard.defaultOutputModes,
      },
      agentCard,
    });
  } catch (err) {
    console.error("Failed to register agent:", err.message);
    res.status(502).json({ error: `Failed to connect to agent: ${err.message}` });
  }
});

/**
 * DELETE /api/agents/:agentId - Remove an agent
 */
app.delete("/api/agents/:agentId", (req, res) => {
  const { agentId } = req.params;

  if (!agents.has(agentId)) {
    return res.status(404).json({ error: "Agent not found" });
  }

  agents.delete(agentId);
  saveAgents();

  res.json({ ok: true });
});

/**
 * POST /api/agents/:agentId/refresh - Refresh agent status and card
 */
app.post("/api/agents/:agentId/refresh", async (req, res) => {
  const { agentId } = req.params;

  const agent = agents.get(agentId);
  if (!agent) {
    return res.status(404).json({ error: "Agent not found" });
  }

  try {
    const client = new A2AClient(agent.url);
    const agentCard = await client.getAgentCard();

    agent.agentCard = agentCard;
    agent.status = "online";
    agent.lastChecked = new Date().toISOString();
    agents.set(agentId, agent);
    saveAgents();

    res.json({
      agent: {
        id: agent.id,
        name: agent.name,
        url: agent.url,
        description: agent.description,
        status: agent.status,
        version: agentCard.version,
        capabilities: agentCard.capabilities,
        skills: agentCard.skills,
        defaultInputModes: agentCard.default_input_modes || agentCard.defaultInputModes,
        defaultOutputModes: agentCard.default_output_modes || agentCard.defaultOutputModes,
      },
    });
  } catch (err) {
    agent.status = "offline";
    agent.lastChecked = new Date().toISOString();
    agents.set(agentId, agent);
    saveAgents();

    res.json({
      agent: {
        id: agent.id,
        name: agent.name,
        url: agent.url,
        description: agent.description,
        status: "offline",
        error: err.message,
      },
    });
  }
});

/**
 * POST /api/agents/fetch-card - Fetch agent card from a URL (preview before registering)
 */
app.post("/api/agents/fetch-card", async (req, res) => {
  const { url } = req.body;

  if (!url) {
    return res.status(400).json({ error: "URL is required" });
  }

  try {
    const client = new A2AClient(url);
    const agentCard = await client.getAgentCard();
    res.json({ agentCard });
  } catch (err) {
    res.status(502).json({ error: `Failed to fetch agent card: ${err.message}` });
  }
});

// =============================================================================
// Quick Chat API (without persistent session)
// =============================================================================

/**
 * POST /api/chat - Send a message to an agent
 */
app.post("/api/chat", async (req, res) => {
  console.log("=== [QuickChat] Received quick chat request ===");
  console.log("[QuickChat] Request body:", JSON.stringify(req.body));
  
  const { agentId, text, systemPrompt, contextId } = req.body;

  if (!agentId) {
    return res.status(400).json({ error: "agentId is required" });
  }
  if (!text) {
    return res.status(400).json({ error: "text is required" });
  }

  const agent = agents.get(agentId);
  if (!agent) {
    return res.status(404).json({ error: "Agent not found" });
  }

  try {
    const client = new A2AClient(agent.url);
    const response = await client.sendMessage({
      text,
      systemPrompt,
      contextId,
    });

    const result = client.extractResult(response);

    res.json({
      contextId: result.contextId || contextId,
      response: {
        text: result.text,
        state: result.state,
        type: "chat",
      },
    });
  } catch (err) {
    console.error("Chat error:", err.message);
    res.status(500).json({ error: `Failed to send message: ${err.message}` });
  }
});

/**
 * POST /api/chat/upload - Upload media and send to an agent
 */
app.post("/api/chat/upload", upload.array("files", 10), async (req, res) => {
  const { agentId, query, systemPrompt, contextId } = req.body;

  if (!agentId) {
    return res.status(400).json({ error: "agentId is required" });
  }
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: "No files provided" });
  }

  const agent = agents.get(agentId);
  if (!agent) {
    return res.status(404).json({ error: "Agent not found" });
  }

  try {
    const images = [];
    let video = null;

    for (const file of req.files) {
      const fileBuffer = fs.readFileSync(file.path);
      const base64 = fileBuffer.toString("base64");

      if (file.mimetype.startsWith("image/")) {
        images.push({
          bytes: base64,
          mimeType: file.mimetype,
        });
      } else if (file.mimetype.startsWith("video/") && !video) {
        video = {
          bytes: base64,
          mimeType: file.mimetype,
        };
      }

      // Clean up uploaded file
      fs.unlinkSync(file.path);
    }

    const client = new A2AClient(agent.url);
    const response = await client.sendMessage({
      text: query || "Analyze the uploaded media.",
      systemPrompt,
      images: images.length > 0 ? images : undefined,
      video,
      contextId,
    });

    const result = client.extractResult(response);

    res.json({
      contextId: result.contextId || contextId,
      response: {
        text: result.text,
        state: result.state,
        type: "media_analysis",
      },
    });
  } catch (err) {
    // Clean up any remaining files
    for (const file of req.files) {
      try {
        fs.unlinkSync(file.path);
      } catch {}
    }
    console.error("Media upload error:", err.message);
    res.status(500).json({ error: `Failed to process media: ${err.message}` });
  }
});

// =============================================================================
// Session API
// =============================================================================

/**
 * GET /api/sessions - List all sessions
 */
app.get("/api/sessions", (req, res) => {
  const sessionList = Array.from(sessions.values()).map((s) => ({
    sessionId: s.sessionId,
    agentId: s.agentId,
    agentName: agents.get(s.agentId)?.name || "Unknown",
    historyLength: s.history.length,
    createdAt: s.createdAt,
  }));
  res.json({ sessions: sessionList });
});

/**
 * POST /api/sessions - Create a new session
 */
app.post("/api/sessions", (req, res) => {
  console.log("=== [Session] Creating new session ===");
  console.log("[Session] Request body:", JSON.stringify(req.body));
  
  const { agentId, systemPrompt } = req.body;

  if (!agentId) {
    console.log("[Session] ERROR: agentId is required");
    return res.status(400).json({ error: "agentId is required" });
  }

  const agent = agents.get(agentId);
  if (!agent) {
    console.log("[Session] ERROR: Agent not found:", agentId);
    return res.status(404).json({ error: "Agent not found" });
  }

  const newSessionId = uuidv4();
  console.log("[Session] Creating session with ID:", newSessionId);
  
  const session = {
    sessionId: newSessionId,
    agentId,
    contextId: null,
    systemPrompt: systemPrompt || "",
    history: [],
    createdAt: new Date().toISOString(),
  };

  sessions.set(session.sessionId, session);
  console.log("[Session] Session created successfully:", session.sessionId);
  console.log("[Session] Total sessions:", sessions.size);

  res.json({ session });
});

/**
 * GET /api/sessions/:sessionId - Get session details
 */
app.get("/api/sessions/:sessionId", (req, res) => {
  const session = sessions.get(req.params.sessionId);
  if (!session) {
    return res.status(404).json({ error: "Session not found" });
  }
  res.json(session);
});

/**
 * POST /api/sessions/:sessionId/chat - Send a message in a session (non-streaming)
 */
app.post("/api/sessions/:sessionId/chat", async (req, res) => {
  console.log("=== [Non-Streaming] Received chat request ===");
  console.log("[Non-Streaming] Session ID:", req.params.sessionId);
  console.log("[Non-Streaming] This is the FALLBACK endpoint - streaming may have failed");
  
  const { text } = req.body;
  const session = sessions.get(req.params.sessionId);

  if (!session) {
    console.log("[Non-Streaming] Session not found");
    return res.status(404).json({ error: "Session not found" });
  }
  if (!text) {
    return res.status(400).json({ error: "text is required" });
  }

  const agent = agents.get(session.agentId);
  if (!agent) {
    return res.status(404).json({ error: "Agent not found" });
  }
  
  console.log("[Non-Streaming] Calling agent:", agent.name, "at", agent.url);

  try {
    // Add user message to history
    session.history.push({
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    });

    const client = new A2AClient(agent.url);
    const response = await client.sendMessage({
      text,
      systemPrompt: session.history.length === 1 ? session.systemPrompt : undefined,
      contextId: session.contextId,
    });

    const result = client.extractResult(response);

    // Update session
    if (result.contextId) {
      session.contextId = result.contextId;
    }
    session.history.push({
      role: "agent",
      content: result.text,
      state: result.state,
      timestamp: new Date().toISOString(),
    });

    res.json({
      contextId: session.contextId,
      response: {
        text: result.text,
        state: result.state,
        type: "chat",
      },
    });
  } catch (err) {
    console.error("Session chat error:", err.message);
    res.status(500).json({ error: `Failed to send message: ${err.message}` });
  }
});

/**
 * POST /api/sessions/:sessionId/chat/stream - Send a message with streaming response (SSE)
 */
app.post("/api/sessions/:sessionId/chat/stream", async (req, res) => {
  console.log("=== [Streaming] Received streaming request ===");
  console.log("[Streaming] Session ID:", req.params.sessionId);
  console.log("[Streaming] Request body:", JSON.stringify(req.body));
  
  const { text } = req.body;
  const session = sessions.get(req.params.sessionId);

  if (!session) {
    console.log("[Streaming] ERROR: Session not found:", req.params.sessionId);
    console.log("[Streaming] Available sessions:", Array.from(sessions.keys()));
    return res.status(404).json({ error: "Session not found" });
  }
  
  console.log("[Streaming] Session found, agentId:", session.agentId);
  console.log("[Streaming] Session contextId:", session.contextId);
  console.log("[Streaming] Session history length:", session.history.length);
  
  if (!text) {
    console.log("[Streaming] ERROR: text is required");
    return res.status(400).json({ error: "text is required" });
  }

  const agent = agents.get(session.agentId);
  if (!agent) {
    console.log("[Streaming] ERROR: Agent not found:", session.agentId);
    return res.status(404).json({ error: "Agent not found" });
  }

  console.log("[Streaming] Agent found:", agent.name, "at", agent.url);
  console.log("[Streaming] Starting SSE stream to agent...");
  
  // Set up SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    // Add user message to history
    session.history.push({
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    });

    const client = new A2AClient(agent.url);
    let finalContent = "";
    let accumulatedTokens = "";
    let finalState = "working";
    const toolCalls = [];

    // Stream events from the agent
    // Note: sendStreamingMessage now returns pre-parsed events
    const isFirstMessage = session.history.length === 1;
    console.log(`[Streaming] First message: ${isFirstMessage}, contextId: ${session.contextId}`);
    
    await client.sendStreamingMessage(
      {
        text,
        systemPrompt: isFirstMessage ? session.systemPrompt : undefined,
        contextId: session.contextId,
      },
      (parsed) => {
        // Update context ID if available
        if (parsed.contextId) {
          session.contextId = parsed.contextId;
        }

        // Build SSE event to send to frontend
        const sseEvent = {
          type: parsed.type,
          content: parsed.content,
          state: parsed.state,
          contextId: session.contextId,
        };

        if (parsed.type === "tool_call" || parsed.type === "tool_start") {
          sseEvent.type = "tool_call";
          sseEvent.toolName = parsed.toolName;
          sseEvent.toolInput = parsed.toolInput;
          toolCalls.push({
            type: "tool_call",
            toolName: parsed.toolName,
            toolInput: parsed.toolInput,
            timestamp: new Date().toISOString(),
          });
          console.log(`[Streaming] Tool call: ${parsed.toolName}`);
        } else if (parsed.type === "tool_result" || parsed.type === "tool_end") {
          sseEvent.type = "tool_result";
          sseEvent.toolName = parsed.toolName;
          sseEvent.toolOutput = parsed.toolOutput;
          toolCalls.push({
            type: "tool_result",
            toolName: parsed.toolName,
            toolOutput: parsed.toolOutput,
            timestamp: new Date().toISOString(),
          });
          console.log(`[Streaming] Tool result: ${parsed.toolName}`);
        } else if (parsed.type === "llm_thought") {
          sseEvent.toolCalls = parsed.toolCalls;
          console.log(`[Streaming] LLM thought with ${(parsed.toolCalls || []).length} tool calls`);
        } else if (parsed.type === "token_done") {
          // Signal the frontend to finalise the current streaming
          // bubble before tool calls appear
          console.log(`[Streaming] Token done — finalise current bubble`);
        } else if (parsed.type === "final_response" || parsed.type === "final") {
          sseEvent.type = "final_response";
          finalContent = parsed.content;
          finalState = parsed.state || "completed";
          console.log(`[Streaming] Final response received`);
        } else if (parsed.type === "token") {
          // Token-by-token streaming — accumulate for history,
          // forward each token individually to the frontend
          if (parsed.content) {
            accumulatedTokens += parsed.content;
          }
          console.log(`[Streaming] Token event: "${parsed.content?.slice(0, 50)}"`);
        } else if (parsed.type === "status") {
          // Generic status messages (e.g. "Processing images…")
          console.log(`[Streaming] Status: ${parsed.content}`);
        }

        res.write(`data: ${JSON.stringify(sseEvent)}\n\n`);
      }
    );

    // Use accumulated tokens as final content if no explicit final event
    if (!finalContent && accumulatedTokens) {
      finalContent = accumulatedTokens;
      finalState = "completed";
    }

    // Add agent response to history
    session.history.push({
      role: "agent",
      content: finalContent,
      state: finalState,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      timestamp: new Date().toISOString(),
    });

    // Send final event
    res.write(`data: ${JSON.stringify({ type: "done", contextId: session.contextId })}\n\n`);
    res.end();
  } catch (err) {
    console.error("Session streaming chat error:", err.message);
    res.write(`data: ${JSON.stringify({ type: "error", error: err.message })}\n\n`);
    res.end();
  }
});

/**
 * POST /api/sessions/:sessionId/upload - Upload media in a session
 */
app.post(
  "/api/sessions/:sessionId/upload",
  upload.array("files", 10),
  async (req, res) => {
    const { query } = req.body;
    const session = sessions.get(req.params.sessionId);

    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "No files provided" });
    }

    const agent = agents.get(session.agentId);
    if (!agent) {
      return res.status(404).json({ error: "Agent not found" });
    }

    try {
      const images = [];
      let video = null;

      for (const file of req.files) {
        const fileBuffer = fs.readFileSync(file.path);
        const base64 = fileBuffer.toString("base64");

        if (file.mimetype.startsWith("image/")) {
          images.push({
            bytes: base64,
            mimeType: file.mimetype,
          });
        } else if (file.mimetype.startsWith("video/") && !video) {
          video = {
            bytes: base64,
            mimeType: file.mimetype,
          };
        }

        fs.unlinkSync(file.path);
      }

      // Add user message to history
      const fileNames = req.files.map((f) => f.originalname).join(", ");
      session.history.push({
        role: "user",
        content: `[Media Upload: ${fileNames}] ${query || ""}`,
        timestamp: new Date().toISOString(),
      });

      const client = new A2AClient(agent.url);
      const response = await client.sendMessage({
        text: query || "Analyze the uploaded media.",
        systemPrompt: session.history.length === 1 ? session.systemPrompt : undefined,
        images: images.length > 0 ? images : undefined,
        video,
        contextId: session.contextId,
      });

      const result = client.extractResult(response);

      // Update session
      if (result.contextId) {
        session.contextId = result.contextId;
      }
      session.history.push({
        role: "agent",
        content: result.text,
        state: result.state,
        timestamp: new Date().toISOString(),
      });

      res.json({
        contextId: session.contextId,
        response: {
          text: result.text,
          state: result.state,
          type: "media_analysis",
        },
      });
    } catch (err) {
      for (const file of req.files) {
        try {
          fs.unlinkSync(file.path);
        } catch {}
      }
      console.error("Session upload error:", err.message);
      res.status(500).json({ error: `Failed to process media: ${err.message}` });
    }
  }
);

/**
 * POST /api/sessions/:sessionId/upload/stream - Upload media with streaming response (SSE)
 */
app.post(
  "/api/sessions/:sessionId/upload/stream",
  upload.array("files", 10),
  async (req, res) => {
    console.log("=== [StreamingUpload] Received streaming upload request ===");
    const { query } = req.body;
    const session = sessions.get(req.params.sessionId);

    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "No files provided" });
    }

    const agent = agents.get(session.agentId);
    if (!agent) {
      return res.status(404).json({ error: "Agent not found" });
    }

    // Set up SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    try {
      const images = [];
      let video = null;

      for (const file of req.files) {
        const fileBuffer = fs.readFileSync(file.path);
        const base64 = fileBuffer.toString("base64");

        if (file.mimetype.startsWith("image/")) {
          images.push({
            bytes: base64,
            mimeType: file.mimetype,
          });
        } else if (file.mimetype.startsWith("video/") && !video) {
          video = {
            bytes: base64,
            mimeType: file.mimetype,
          };
        }

        fs.unlinkSync(file.path);
      }

      // Add user message to history
      const fileNames = req.files.map((f) => f.originalname).join(", ");
      session.history.push({
        role: "user",
        content: `[Media Upload: ${fileNames}] ${query || ""}`,
        timestamp: new Date().toISOString(),
      });

      const client = new A2AClient(agent.url);
      let finalContent = "";
      let accumulatedTokens = "";
      let finalState = "working";
      const toolCalls = [];

      // Stream events from the agent
      console.log("[StreamingUpload] Starting streaming to agent:", agent.url);
      await client.sendStreamingMessage(
        {
          text: query || "Analyze the uploaded media.",
          systemPrompt: session.history.length === 1 ? session.systemPrompt : undefined,
          images: images.length > 0 ? images : undefined,
          video,
          contextId: session.contextId,
        },
        (parsed) => {
          // Update context ID if available
          if (parsed.contextId) {
            session.contextId = parsed.contextId;
          }

          // Build SSE event to send to frontend
          const sseEvent = {
            type: parsed.type,
            content: parsed.content,
            state: parsed.state,
            contextId: session.contextId,
          };

          if (parsed.type === "tool_call" || parsed.type === "tool_start") {
            sseEvent.type = "tool_call";
            sseEvent.toolName = parsed.toolName;
            sseEvent.toolInput = parsed.toolInput;
            toolCalls.push({
              type: "tool_call",
              toolName: parsed.toolName,
              toolInput: parsed.toolInput,
              timestamp: new Date().toISOString(),
            });
            console.log(`[StreamingUpload] Tool call: ${parsed.toolName}`);
          } else if (parsed.type === "tool_result" || parsed.type === "tool_end") {
            sseEvent.type = "tool_result";
            sseEvent.toolName = parsed.toolName;
            sseEvent.toolOutput = parsed.toolOutput;
            toolCalls.push({
              type: "tool_result",
              toolName: parsed.toolName,
              toolOutput: parsed.toolOutput,
              timestamp: new Date().toISOString(),
            });
            console.log(`[StreamingUpload] Tool result: ${parsed.toolName}`);
          } else if (parsed.type === "llm_thought") {
            sseEvent.toolCalls = parsed.toolCalls;
            console.log(`[StreamingUpload] LLM thought with ${(parsed.toolCalls || []).length} tool calls`);
          } else if (parsed.type === "token_done") {
            // Signal the frontend to finalise the current streaming
            // bubble before tool calls appear
            console.log(`[StreamingUpload] Token done — finalise current bubble`);
          } else if (parsed.type === "final_response" || parsed.type === "final") {
            sseEvent.type = "final_response";
            finalContent = parsed.content;
            finalState = parsed.state || "completed";
            console.log(`[StreamingUpload] Final response received`);
          } else if (parsed.type === "token") {
            // Token-by-token streaming — accumulate for history
            // and forward to frontend for real-time display
            if (parsed.content) {
              accumulatedTokens += parsed.content;
            }
            console.log(`[StreamingUpload] Token event: "${parsed.content?.slice(0, 50)}"`);
          } else if (parsed.type === "status") {
            console.log(`[StreamingUpload] Status: ${parsed.content}`);
          }

          res.write(`data: ${JSON.stringify(sseEvent)}\n\n`);
        }
      );

      // Use accumulated tokens as final content if no explicit final event
      if (!finalContent && accumulatedTokens) {
        finalContent = accumulatedTokens;
        finalState = "completed";
      }

      // Add agent response to history
      session.history.push({
        role: "agent",
        content: finalContent,
        state: finalState,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        timestamp: new Date().toISOString(),
      });

      // Send final event
      res.write(`data: ${JSON.stringify({ type: "done", contextId: session.contextId })}\n\n`);
      res.end();
    } catch (err) {
      for (const file of req.files || []) {
        try {
          fs.unlinkSync(file.path);
        } catch {}
      }
      console.error("Streaming upload error:", err.message);
      res.write(`data: ${JSON.stringify({ type: "error", error: err.message })}\n\n`);
      res.end();
    }
  }
);

// Catch-all for SPA in production
if (process.env.NODE_ENV === "production") {
  app.get("*", (req, res) => {
    const clientDistPath = path.join(__dirname, "client", "dist", "index.html");
    if (fs.existsSync(clientDistPath)) {
      res.sendFile(clientDistPath);
    } else {
      const altPath = path.join(__dirname, "..", "client", "dist", "index.html");
      if (fs.existsSync(altPath)) {
        res.sendFile(altPath);
      } else {
        res.status(404).json({
          error: "Frontend not built. Run 'npm run build' in the client directory first.",
          hint: "In development, run the client dev server separately on port 5173"
        });
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`A2A Multi-Agent Chat Server running on http://localhost:${PORT}`);
  console.log(`Registered agents: ${agents.size}`);
});
