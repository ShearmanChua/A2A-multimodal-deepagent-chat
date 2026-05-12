/**
 * A2A Protocol Client for communicating with A2A-compatible agents.
 *
 * Implements the JSON-RPC based A2A protocol:
 * - Fetches agent card from /.well-known/agent.json
 * - Sends messages via message/send (JSON-RPC)
 */

const { v4: uuidv4 } = require("uuid");

// Default timeout for A2A requests (5 minutes for long-running tasks)
const DEFAULT_TIMEOUT = parseInt(process.env.A2A_TIMEOUT || "300000", 10);

class A2AClient {
  constructor(baseUrl, options = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); // Remove trailing slash
    this.timeout = options.timeout || DEFAULT_TIMEOUT;
  }

  /**
   * Fetch with timeout support.
   * @param {string} url - URL to fetch
   * @param {object} options - Fetch options
   * @returns {Promise<Response>}
   */
  async fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      return response;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error(`Request timed out after ${this.timeout}ms`);
      }
      // Provide more helpful error messages
      if (err.cause?.code === "ECONNREFUSED") {
        throw new Error(`Connection refused: Cannot connect to ${url}. Is the agent running?`);
      }
      if (err.cause?.code === "ENOTFOUND") {
        throw new Error(`Host not found: ${url}. Check the agent URL.`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Fetch the agent card from the A2A server.
   * @returns {Promise<object>} The agent card JSON.
   */
  async getAgentCard() {
    const url = `${this.baseUrl}/.well-known/agent.json`;
    const res = await this.fetchWithTimeout(url, {
      headers: { Accept: "application/json" },
    });

    if (!res.ok) {
      throw new Error(`Failed to fetch agent card: ${res.status} ${res.statusText}`);
    }

    return res.json();
  }

  /**
   * Build an A2A message payload.
   *
   * @param {object} opts
   * @param {string} opts.text - User text query.
   * @param {string} [opts.systemPrompt] - Optional system prompt.
   * @param {Array<{bytes: string, mimeType: string}>} [opts.images] - Base64 images.
   * @param {{bytes: string, mimeType: string}} [opts.video] - Base64 video.
   * @param {Array<{bytes: string, mimeType: string, name: string}>} [opts.documents] - Base64 documents.
   * @param {string} [opts.contextId] - Conversation context ID.
   * @returns {object} JSON-RPC request body for message/send.
   */
  buildSendMessageRequest({ text, systemPrompt, images, video, documents, contextId }) {
    const parts = [];

    // User text
    parts.push({ kind: "text", text });

    // Images (multiple)
    if (images && images.length > 0) {
      for (const img of images) {
        parts.push({
          kind: "file",
          file: { bytes: img.bytes, mimeType: img.mimeType || "image/jpeg" },
        });
      }
    }

    // Video (single)
    if (video) {
      parts.push({
        kind: "file",
        file: { bytes: video.bytes, mimeType: video.mimeType || "video/mp4" },
      });
    }

    // Documents — FileWithUri (presigned URL) when the object store uploaded the
    // file; FileWithBytes (base64) as fallback when the object store is not
    // available.  Original filenames go in metadata so the agent executor can
    // map them to /uploads/<name> in the virtual filesystem.
    if (documents && documents.length > 0) {
      for (const doc of documents) {
        if (doc.presignedUrl) {
          parts.push({
            kind: "file",
            file: { uri: doc.presignedUrl, mimeType: doc.mimeType },
          });
        } else {
          parts.push({
            kind: "file",
            file: { bytes: doc.bytes, mimeType: doc.mimeType },
          });
        }
      }
    }

    const messageId = uuidv4().replace(/-/g, "");

    // Build metadata: always include system_prompt and document_names when present
    const metadata = {};
    if (systemPrompt) metadata.system_prompt = systemPrompt;
    if (documents && documents.length > 0) {
      metadata.document_names = documents.map((d) => d.name);
    }

    return {
      jsonrpc: "2.0",
      id: uuidv4(),
      method: "message/send",
      params: {
        message: {
          role: "user",
          parts,
          messageId,
          ...(contextId ? { contextId } : {}),
        },
        ...(Object.keys(metadata).length > 0 ? { metadata } : {}),
      },
    };
  }

  /**
   * Build a streaming message request.
   * @param {object} opts - Same as buildSendMessageRequest options.
   * @returns {object} JSON-RPC request body for message/stream.
   */
  buildStreamMessageRequest(opts) {
    const req = this.buildSendMessageRequest(opts);
    req.method = "message/stream";
    return req;
  }

  /**
   * Send a message to the A2A agent and return the full response.
   *
   * @param {object} opts - Same as buildSendMessageRequest options.
   * @returns {Promise<object>} The JSON-RPC response.
   */
  async sendMessage(opts) {
    const body = this.buildSendMessageRequest(opts);

    console.log(`[A2AClient] Sending message to ${this.baseUrl}`);

    const res = await this.fetchWithTimeout(this.baseUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`A2A request failed: ${res.status} — ${text}`);
    }

    return res.json();
  }

  /**
   * Send a streaming message to the A2A agent.
   * Yields events as they arrive via Server-Sent Events.
   *
   * @param {object} opts - Same as buildSendMessageRequest options.
   * @param {function} onEvent - Callback for each SSE event.
   * @returns {Promise<object>} The final accumulated response.
   */
  async sendStreamingMessage(opts, onEvent) {
    const body = this.buildStreamMessageRequest(opts);

    console.log(`[A2AClient] Sending streaming message to ${this.baseUrl}`);
    console.log(`[A2AClient] Request body:`, JSON.stringify(body, null, 2));

    const res = await this.fetchWithTimeout(this.baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`A2A streaming request failed: ${res.status} — ${text}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let lastEvent = null;
    let contextId = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events - handle both "data: " prefix and raw JSON
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      let currentData = "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          currentData += line.slice(6);
        } else if (line.trim() && !line.startsWith(":")) {
          // Some SSE implementations send raw JSON without "data: " prefix
          currentData += line;
        }
        
        if ((line === "" || line.trim() === "") && currentData) {
          try {
            const event = JSON.parse(currentData);
            console.log(`[A2AClient] Received SSE event:`, JSON.stringify(event).slice(0, 200));
            
            // Extract contextId from the event
            const result = event?.result || event;
            if (result?.contextId) {
              contextId = result.contextId;
            }
            
            // Parse the event and add contextId
            const parsed = this.parseStreamEvent(event);
            parsed.contextId = contextId;
            
            lastEvent = parsed;
            if (onEvent) onEvent(parsed);
          } catch (e) {
            // Not valid JSON, skip
            console.log(`[A2AClient] Failed to parse SSE data:`, currentData.slice(0, 100), e.message);
          }
          currentData = "";
        }
      }
    }

    return lastEvent;
  }

  /**
   * Extract streaming events and parse tool calls.
   *
   * Prefers structured `metadata` on the message or event when available
   * (set by the agent executor), falling back to regex-based text parsing
   * for backward compatibility with older agents.
   *
   * Metadata schema (on `message.metadata` or `event.metadata`):
   *   - token:          { event_type: "token" }
   *   - tool_call:      { event_type: "tool_call", tool_name, tool_input }
   *   - llm_thought:    { event_type: "llm_thought", tool_calls: [...] }
   *   - tool_result:    { event_type: "tool_result", tool_name, tool_output }
   *   - final_response: { event_type: "final_response" }
   *   - status:         { event_type: "status" }
   *
   * @param {object} event - SSE event data.
   * @returns {{ type: string, content: string, toolName?: string, toolInput?: string, toolOutput?: string, toolCalls?: Array, state?: string }}
   */
  parseStreamEvent(event) {
    const result = event?.result || event;
    
    console.log("[A2AClient.parseStreamEvent] Raw event:", JSON.stringify(result).slice(0, 300));

    const state = result?.status?.state || result?.state;

    // ── Try metadata-based detection ──────────────────────────
    // The metadata can be at different paths depending on the A2A implementation:
    // - result.status.message.metadata (message-level metadata)
    // - result.status.metadata (status-level metadata from updater.update_status)
    // - result.metadata (top-level metadata)
    const msgMeta = result?.status?.message?.metadata
      || result?.status?.metadata
      || result?.metadata
      || null;

    if (msgMeta?.event_type) {
      const evtType = msgMeta.event_type;

      // Extract text content from message parts
      let text = "";
      if (result?.status?.message?.parts) {
        text = result.status.message.parts
          .filter((p) => p.kind === "text" || p.text || p.root?.kind === "text" || p.root?.text)
          .map((p) => p.text || p.root?.text || "")
          .join("\n");
      }

      console.log("[A2AClient.parseStreamEvent] Metadata event_type:", evtType, "content:", text.slice(0, 100));

      switch (evtType) {
        case "token": {
          // Strip the backward-compat "[token] " prefix if present
          const tokenContent = text.replace(/^\[token\] /, "");
          console.log("[A2AClient.parseStreamEvent] Detected token:", tokenContent.slice(0, 100));
          return { type: "token", content: tokenContent, state };
        }

        case "final_response": {
          // This is the actual final response
          console.log("[A2AClient.parseStreamEvent] Detected final_response:", text.slice(0, 100));
          return { type: "final_response", content: text, state };
        }

        case "tool_call": {
          const toolName = msgMeta.tool_name || "";
          const toolInput = msgMeta.tool_input || "";
          // Strip the backward-compat prefix and code block from display text
          const cleanContent = text
            .replace(/\[tool_start:[^\]]+\]\s*/, "")
            .replace(/```json\n[\s\S]*?\n```/, "")
            .trim();
          console.log("[A2AClient.parseStreamEvent] Detected tool_call:", toolName);
          return {
            type: "tool_call",
            content: cleanContent,
            toolName,
            toolInput,
            state,
          };
        }

        case "tool_result": {
          const toolName = msgMeta.tool_name || "";
          const toolOutput = msgMeta.tool_output || "";
          const cleanContent = text
            .replace(/\[tool_end:[^\]]+\]\s*/, "")
            .replace(/```\n[\s\S]*?\n```/, "")
            .trim();
          console.log("[A2AClient.parseStreamEvent] Detected tool_result:", toolName);
          return {
            type: "tool_result",
            content: cleanContent,
            toolName,
            toolOutput,
            state,
          };
        }

        case "llm_thought": {
          const toolCalls = msgMeta.tool_calls || [];
          console.log("[A2AClient.parseStreamEvent] Detected llm_thought, tool_calls:", toolCalls.length);
          return {
            type: "llm_thought",
            content: text,
            toolCalls,
            state,
          };
        }

        case "input_required":
          return { type: "input_required", content: text, state };

        case "status":
        default:
          return { type: "status", content: text, state };
      }
    }

    // ── Fallback: regex-based text parsing (backward compat) ────────
    if (result?.status?.message?.parts) {
      const text = result.status.message.parts
        .filter((p) => p.kind === "text" || p.text || p.root?.kind === "text" || p.root?.text)
        .map((p) => p.text || p.root?.text || "")
        .join("\n");
      
      console.log("[A2AClient.parseStreamEvent] Fallback text parsing:", text.slice(0, 200));
      
      // Parse token-by-token streaming marker: "[token] <content>"
      const tokenMatch = text.match(/^\[token\] ([\s\S]*)$/);
      if (tokenMatch) {
        return {
          type: "token",
          content: tokenMatch[1],
          state: result.status.state,
        };
      }
      
      // Parse token_done marker
      if (text.trim() === "[token_done]") {
        return {
          type: "token_done",
          content: "",
          state: result.status.state,
        };
      }
      
      // Parse tool call markers
      const toolStartMatch = text.match(/\[tool_start:([^\]]+)\]/);
      const toolEndMatch = text.match(/\[tool_end:([^\]]+)\]/);
      
      if (toolStartMatch) {
        const toolName = toolStartMatch[1];
        const content = text.replace(/\[tool_start:[^\]]+\]\s*/, "");
        const inputMatch = text.match(/```json\n([\s\S]*?)\n```/);
        console.log("[A2AClient.parseStreamEvent] Fallback detected tool_start:", toolName);
        return {
          type: "tool_call",
          content,
          toolName,
          toolInput: inputMatch ? inputMatch[1] : "",
          state: result.status.state,
        };
      }
      
      if (toolEndMatch) {
        const toolName = toolEndMatch[1];
        const content = text.replace(/\[tool_end:[^\]]+\]\s*/, "");
        const outputMatch = text.match(/```\n([\s\S]*?)\n```/);
        console.log("[A2AClient.parseStreamEvent] Fallback detected tool_end:", toolName);
        return {
          type: "tool_result",
          content,
          toolName,
          toolOutput: outputMatch ? outputMatch[1] : "",
          state: result.status.state,
        };
      }
      
      // Generic status message
      return {
        type: "status",
        content: text,
        state: result.status.state,
      };
    }
    
    // Check for artifacts (final response)
    if (result?.artifacts && result.artifacts.length > 0) {
      let text = "";
      for (const artifact of result.artifacts) {
        if (artifact.parts) {
          const artifactText = artifact.parts
            .filter((p) => p.kind === "text" || p.text || p.root?.kind === "text" || p.root?.text)
            .map((p) => p.text || p.root?.text || "")
            .join("\n");
          if (artifactText) {
            text = text ? `${text}\n\n${artifactText}` : artifactText;
          }
        }
      }
      return {
        type: "final_response",
        content: text,
        state: result.status?.state || "completed",
      };
    }
    
    return {
      type: "unknown",
      content: JSON.stringify(result),
      state: "unknown",
    };
  }

  /**
   * Extract the text content and task state from an A2A response.
   *
   * @param {object} response - The JSON-RPC response.
   * @returns {{ text: string, state: string, taskId: string, contextId: string }}
   */
  extractResult(response) {
    const result = response?.result || response;

    // Handle task result
    if (result?.status?.state) {
      const state = result.status.state;
      const taskId = result.id || "";
      const contextId = result.contextId || "";

      // Extract text from status message
      let text = "";
      if (result.status?.message?.parts) {
        text = result.status.message.parts
          .filter((p) => p.kind === "text" || p.text || p.root?.kind === "text" || p.root?.text)
          .map((p) => p.text || p.root?.text || "")
          .join("\n");
      }

      // Extract text from artifacts
      if (result.artifacts && result.artifacts.length > 0) {
        for (const artifact of result.artifacts) {
          if (artifact.parts) {
            const artifactText = artifact.parts
              .filter((p) => p.kind === "text" || p.text || p.root?.kind === "text" || p.root?.text)
              .map((p) => p.text || p.root?.text || "")
              .join("\n");
            if (artifactText) {
              text = text ? `${text}\n\n${artifactText}` : artifactText;
            }
          }
        }
      }

      return { text, state, taskId, contextId };
    }

    // Fallback
    return {
      text: JSON.stringify(result, null, 2),
      state: "unknown",
      taskId: "",
      contextId: "",
    };
  }
}

module.exports = { A2AClient };
