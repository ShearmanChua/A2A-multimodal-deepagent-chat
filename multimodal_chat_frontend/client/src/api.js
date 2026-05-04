/**
 * API client for the Multimodal Chat Frontend.
 * 
 * Handles:
 * - Agent registry management (list, register, remove agents)
 * - A2A protocol communication with selected agent
 * - Media upload and chat functionality
 */

const API_BASE = "/api";

/**
 * Generic fetch wrapper with error handling.
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(errorData.error || errorData.details || res.statusText);
  }

  return res.json();
}

// =============================================================================
// Agent Registry API
// =============================================================================

/**
 * List all registered agents.
 * @returns {Promise<{agents: Array<{id, name, url, description, status, capabilities}>}>}
 */
export const listAgents = () => apiFetch("/agents");

/**
 * Register a new agent.
 * @param {object} agent - Agent details
 * @param {string} agent.name - Display name
 * @param {string} agent.url - A2A agent URL (e.g., http://localhost:10010)
 * @param {string} [agent.description] - Optional description
 * @returns {Promise<{agent: object, agentCard: object}>}
 */
export const registerAgent = (agent) =>
  apiFetch("/agents", {
    method: "POST",
    body: JSON.stringify(agent),
  });

/**
 * Remove a registered agent.
 * @param {string} agentId - Agent ID to remove
 */
export const removeAgent = (agentId) =>
  apiFetch(`/agents/${agentId}`, { method: "DELETE" });

/**
 * Refresh agent status and card.
 * @param {string} agentId - Agent ID to refresh
 */
export const refreshAgent = (agentId) =>
  apiFetch(`/agents/${agentId}/refresh`, { method: "POST" });

/**
 * Get agent card directly from an A2A agent URL.
 * @param {string} agentUrl - The A2A agent URL
 */
export const fetchAgentCard = (agentUrl) =>
  apiFetch("/agents/fetch-card", {
    method: "POST",
    body: JSON.stringify({ url: agentUrl }),
  });

// =============================================================================
// Chat Session API
// =============================================================================

/**
 * Health check.
 */
export const healthCheck = () => apiFetch("/health");

/**
 * Create a new chat session with a specific agent.
 * @param {string} agentId - The agent to chat with
 * @param {string} [systemPrompt] - Optional system prompt
 */
export const createSession = (agentId, systemPrompt) =>
  apiFetch("/sessions", {
    method: "POST",
    body: JSON.stringify({ agentId, systemPrompt }),
  });

/**
 * Get session info.
 */
export const getSession = (sessionId) => apiFetch(`/sessions/${sessionId}`);

/**
 * List all sessions.
 */
export const listSessions = () => apiFetch("/sessions");

/**
 * Send a text message to the agent in a session (non-streaming).
 * @param {string} sessionId - Session ID
 * @param {string} text - Message text
 */
export const sendMessage = (sessionId, text) =>
  apiFetch(`/sessions/${sessionId}/chat`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });

/**
 * Send a streaming message to the agent in a session.
 * Returns an async generator that yields events as they arrive.
 *
 * @param {string} sessionId - Session ID
 * @param {string} text - Message text
 * @param {function} onEvent - Callback for each event
 * @returns {Promise<void>}
 */
export async function sendStreamingMessage(sessionId, text, onEvent) {
  console.log("[API] ========================================");
  console.log("[API] Sending streaming message to session:", sessionId);
  console.log("[API] Session ID type:", typeof sessionId);
  console.log("[API] Text:", text?.substring(0, 50));
  console.log("[API] Streaming URL:", `${API_BASE}/sessions/${sessionId}/chat/stream`);
  console.log("[API] ========================================");
  
  if (!sessionId) {
    console.error("[API] ERROR: sessionId is null or undefined!");
    throw new Error("sessionId is required for streaming");
  }
  
  let res;
  try {
    console.log("[API] Making fetch request...");
    res = await fetch(`${API_BASE}/sessions/${sessionId}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    console.log("[API] Fetch completed, status:", res.status);
  } catch (fetchErr) {
    console.error("[API] Streaming fetch failed:", fetchErr);
    console.error("[API] Fetch error name:", fetchErr.name);
    console.error("[API] Fetch error message:", fetchErr.message);
    throw fetchErr;
  }

  console.log("[API] Streaming response status:", res.status);
  console.log("[API] Response headers:", Object.fromEntries(res.headers.entries()));
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    console.error("[API] Streaming response error:", err);
    throw new Error(err.error || err.details || res.statusText);
  }

  console.log("[API] Response OK, starting to read stream...");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventCount = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      console.log("[API] Stream done, total events:", eventCount);
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentData = "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        currentData += line.slice(6);
      } else if (line === "" && currentData) {
        try {
          const event = JSON.parse(currentData);
          eventCount++;
          console.log("[API] Parsed event #" + eventCount + ":", event.type);
          if (onEvent) onEvent(event);
        } catch (parseErr) {
          console.warn("[API] Failed to parse SSE data:", currentData.substring(0, 100));
        }
        currentData = "";
      }
    }
  }
  
  console.log("[API] Streaming complete, processed", eventCount, "events");
}

/**
 * Upload media files and send to the agent.
 * @param {string} sessionId - Session ID
 * @param {FileList|File[]} files - Files to upload
 * @param {string} [query] - Optional query text
 */
export async function uploadMedia(sessionId, files, query) {
  const formData = new FormData();
  
  for (const file of files) {
    formData.append("files", file);
  }
  
  if (query) formData.append("query", query);

  const res = await fetch(`${API_BASE}/sessions/${sessionId}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || err.details || res.statusText);
  }

  return res.json();
}

/**
 * Upload media files with streaming response for tool call display.
 * @param {string} sessionId - Session ID
 * @param {FileList|File[]} files - Files to upload
 * @param {string} [query] - Optional query text
 * @param {function} onEvent - Callback for each event
 */
export async function uploadMediaStreaming(sessionId, files, query, onEvent) {
  console.log("[API] ========================================");
  console.log("[API] Sending streaming upload to session:", sessionId);
  console.log("[API] Files:", Array.from(files).map(f => f.name).join(", "));
  console.log("[API] Query:", query?.substring(0, 50));
  console.log("[API] ========================================");
  
  const formData = new FormData();
  
  for (const file of files) {
    formData.append("files", file);
  }
  
  if (query) formData.append("query", query);

  let res;
  try {
    res = await fetch(`${API_BASE}/sessions/${sessionId}/upload/stream`, {
      method: "POST",
      body: formData,
    });
  } catch (fetchErr) {
    console.error("[API] Streaming upload fetch failed:", fetchErr);
    throw fetchErr;
  }

  console.log("[API] Streaming upload response status:", res.status);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    console.error("[API] Streaming upload error:", err);
    throw new Error(err.error || err.details || res.statusText);
  }

  console.log("[API] Response OK, starting to read stream...");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventCount = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      console.log("[API] Stream done, total events:", eventCount);
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentData = "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        currentData += line.slice(6);
      } else if (line === "" && currentData) {
        try {
          const event = JSON.parse(currentData);
          eventCount++;
          console.log("[API] Parsed upload event #" + eventCount + ":", event.type);
          if (onEvent) onEvent(event);
        } catch (parseErr) {
          console.warn("[API] Failed to parse SSE data:", currentData.substring(0, 100));
        }
        currentData = "";
      }
    }
  }
  
  console.log("[API] Streaming upload complete, processed", eventCount, "events");
}

// =============================================================================
// Quick Chat API (without session)
// =============================================================================

/**
 * Send a quick message to an agent without creating a persistent session.
 * @param {string} agentId - Agent ID
 * @param {string} text - Message text
 * @param {string} [systemPrompt] - Optional system prompt
 */
export const quickChat = (agentId, text, systemPrompt) =>
  apiFetch("/chat", {
    method: "POST",
    body: JSON.stringify({ agentId, text, systemPrompt }),
  });

/**
 * Quick media upload to an agent.
 * @param {string} agentId - Agent ID
 * @param {FileList|File[]} files - Files to upload
 * @param {string} [query] - Optional query text
 * @param {string} [systemPrompt] - Optional system prompt
 */
export async function quickMediaUpload(agentId, files, query, systemPrompt) {
  const formData = new FormData();
  
  formData.append("agentId", agentId);
  
  for (const file of files) {
    formData.append("files", file);
  }
  
  if (query) formData.append("query", query);
  if (systemPrompt) formData.append("systemPrompt", systemPrompt);

  const res = await fetch(`${API_BASE}/chat/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || err.details || res.statusText);
  }

  return res.json();
}
