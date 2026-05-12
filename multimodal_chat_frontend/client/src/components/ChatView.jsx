import { useState, useRef, useEffect, useMemo } from "react";
import {
  Bot,
  Brain,
  Send,
  Image,
  Video,
  FileText,
  Paperclip,
  X,
  Loader2,
  Settings,
  Zap,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Trash2,
  RotateCcw,
  Wrench,
  CheckCircle,
  Play,
  Sparkles,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

// ---------------------------------------------------------------------------
// Helpers — classify & group messages
// ---------------------------------------------------------------------------

/** Returns true when the message type is a tool event (call or result). */
function isToolType(type) {
  return [
    "tool_call", "tool_start",
    "tool_result", "tool_end",
  ].includes(type);
}

/** Returns true when the message type is a thought event. */
function isThoughtType(type) {
  return type === "llm_thought";
}

/** Returns true when the message type is any tool/thought event. */
function isToolThinkingType(type) {
  return isToolType(type) || isThoughtType(type);
}

/**
 * Returns true when the message is an assistant text/streaming bubble
 * (i.e. the LLM's intermediate output that may be "thoughts").
 */
function isAssistantText(msg) {
  return (
    msg.role === "assistant" &&
    ["text", "streaming", "media_analysis"].includes(msg.type)
  );
}

/**
 * Group messages so that thoughts and tools are in **separate** collapsible
 * sections.  Returns an array of items:
 *   - `{ kind: "message", message }` — a regular message
 *   - `{ kind: "thought", messages: [...], id }` — collapsible thought group
 *   - `{ kind: "tools",   messages: [...], id }` — collapsible tools group
 */
function groupMessages(messages) {
  const groups = [];
  let currentGroup = null; // { kind, messages, id }
  let deferredFinalResponse = null; // Hold streaming/media_analysis messages to place after tool groups

  /** Flush the current group into `groups` if it exists. */
  function flush() {
    if (currentGroup) {
      groups.push(currentGroup);
      currentGroup = null;
    }
  }

  /**
   * Peek ahead: is there a tool event anywhere after index `i`
   * (before the next user message)?
   */
  function hasToolEventAfter(i) {
    for (let j = i + 1; j < messages.length; j++) {
      const m = messages[j];
      if (m.role === "user") return false;
      if (isToolType(m.type)) return true;
    }
    return false;
  }

  /**
   * Peek ahead: is there a tool/thought event anywhere after index `i`
   * (before the next user message)?
   */
  function hasToolOrThoughtAfter(i) {
    for (let j = i + 1; j < messages.length; j++) {
      const m = messages[j];
      if (m.role === "user") return false;
      if (isToolType(m.type) || isThoughtType(m.type)) return true;
    }
    return false;
  }

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    // ── Thought event (llm_thought) → goes into a "thought" group ──
    if (isThoughtType(msg.type)) {
      // If we were building a tools group, flush it first
      if (currentGroup && currentGroup.kind !== "thought") flush();
      if (!currentGroup) {
        currentGroup = { kind: "thought", messages: [], id: `thought-${msg.id}` };
      }
      currentGroup.messages.push(msg);
      continue;
    }

    // ── Tool event (tool_call, tool_result) → goes into a "tools" group ──
    if (isToolType(msg.type)) {
      // If we were building a thought group, flush it first
      if (currentGroup && currentGroup.kind !== "tools") flush();
      if (!currentGroup) {
        currentGroup = { kind: "tools", messages: [], id: `tools-${msg.id}` };
      }
      currentGroup.messages.push(msg);
      continue;
    }

    // ── Streaming/media_analysis messages: defer if there are tool events after ──
    // This ensures the final response appears AFTER the tool groups, not before.
    if ((msg.type === "streaming" || msg.type === "media_analysis") && hasToolOrThoughtAfter(i)) {
      // Defer this message to be added after the tool/thought groups
      deferredFinalResponse = msg;
      continue;
    }

    // ── Assistant text followed by more tool events → absorb as thought ──
    // BUT: Don't absorb these message types as thoughts:
    // - "streaming" = real-time streaming feedback (should always be visible)
    // - "media_analysis" = final response for media uploads
    // - "text" at the end of the conversation = final response
    // The key insight: if there are no tool events AFTER this message,
    // it's likely the final response and should be displayed normally.
    const shouldAbsorbAsThought =
      isAssistantText(msg) &&
      msg.type !== "streaming" &&
      msg.type !== "media_analysis" &&
      hasToolEventAfter(i);
    
    if (shouldAbsorbAsThought) {
      if (currentGroup && currentGroup.kind !== "thought") flush();
      if (!currentGroup) {
        currentGroup = { kind: "thought", messages: [], id: `thought-${msg.id}` };
      }
      currentGroup.messages.push({ ...msg, _isIntermediateThought: true });
      continue;
    }

    // ── Anything else (user msg, final assistant text, errors, …) ──
    flush();
    
    // If we have a deferred final response and this is a user message,
    // add the deferred response before the user message
    if (deferredFinalResponse && msg.role === "user") {
      groups.push({ kind: "message", message: deferredFinalResponse });
      deferredFinalResponse = null;
    }
    
    groups.push({ kind: "message", message: msg });
  }

  flush();
  
  // Add any deferred final response at the end (after all tool/thought groups)
  if (deferredFinalResponse) {
    groups.push({ kind: "message", message: deferredFinalResponse });
  }
  
  return groups;
}

/**
 * Extract images from tool output string.
 * Looks for base64 data URLs, http URLs, and JSON with image data.
 */
function extractImagesFromOutput(output) {
  if (!output) return { images: [], text: String(output || "") };
  
  // Convert to string if not already
  const outputStr = typeof output === "string" ? output : JSON.stringify(output);
  
  console.log("[extractImages] Input length:", outputStr.length);
  console.log("[extractImages] First 300 chars:", outputStr.substring(0, 300));
  
  const images = [];
  let text = outputStr;
  
  // Look for base64 image data URLs (already formatted)
  const base64DataUrlRegex = /data:image\/[^;]+;base64,[A-Za-z0-9+/=]+/g;
  const base64DataUrlMatches = outputStr.match(base64DataUrlRegex);
  if (base64DataUrlMatches) {
    base64DataUrlMatches.forEach((match) => {
      images.push(match);
      text = text.replace(match, "[Image]");
    });
  }
  
  // Look for http(s) image URLs
  const urlRegex = /https?:\/\/[^\s"']+\.(jpg|jpeg|png|gif|webp|bmp)(\?[^\s"']*)?/gi;
  const urlMatches = outputStr.match(urlRegex);
  if (urlMatches) {
    urlMatches.forEach((match) => {
      if (!images.includes(match)) {
        images.push(match);
      }
    });
  }
  
  // Look for Python-style dict with 'base64' field: {'type': 'image', 'base64': '...'}
  // The base64 data starts with /9j/ for JPEG or iVBOR for PNG
  // Match until we hit a quote, bracket, or end of string
  // The data may be truncated with … or ...
  const pythonBase64Regex = /'base64':\s*'(\/9j\/[A-Za-z0-9+/=]+)/g;
  let pythonMatch;
  while ((pythonMatch = pythonBase64Regex.exec(outputStr)) !== null) {
    let b64 = pythonMatch[1];
    console.log("[extractImages] Found Python base64, length:", b64.length);
    // Use any base64 data that's at least 100 chars (even if truncated, we can try to display)
    if (b64.length > 100) {
      images.push(`data:image/jpeg;base64,${b64}`);
      text = text.replace(pythonMatch[0], "'base64': '[Image]'");
      console.log("[extractImages] Added image with base64 length:", b64.length);
    }
  }
  
  // Also look for JSON-style with double quotes: "base64": "..."
  const jsonBase64Regex = /"base64":\s*"(\/9j\/[A-Za-z0-9+/=]+)/g;
  let jsonMatch;
  while ((jsonMatch = jsonBase64Regex.exec(outputStr)) !== null) {
    let b64 = jsonMatch[1];
    console.log("[extractImages] Found JSON base64, length:", b64.length);
    if (b64.length > 1000) {
      images.push(`data:image/jpeg;base64,${b64}`);
      text = text.replace(jsonMatch[0], '"base64": "[Image]"');
    }
  }
  
  // Look for raw base64 JPEG data (starts with /9j/)
  // This is a more aggressive match for cases where the base64 is inline
  if (images.length === 0) {
    const rawJpegRegex = /\/9j\/[A-Za-z0-9+/=]{1000,}/g;
    let rawMatch;
    while ((rawMatch = rawJpegRegex.exec(outputStr)) !== null) {
      const b64 = rawMatch[0];
      console.log("[extractImages] Found raw JPEG base64, length:", b64.length);
      images.push(`data:image/jpeg;base64,${b64}`);
      text = text.replace(b64, "[Image]");
    }
  }
  
  console.log("[extractImages] Total images found:", images.length);
  
  // Truncate text if we found images
  if (images.length > 0) {
    // Show a summary instead of the full output
    text = `[${images.length} image(s) extracted]`;
  }
  
  return { images, text };
}

/**
 * Display tool output with image/video support.
 */
function ToolOutputDisplay({ output }) {
  console.log("[ToolOutputDisplay] Raw output:", typeof output, output?.substring?.(0, 200) || output);
  const { images, text } = extractImagesFromOutput(output);
  console.log("[ToolOutputDisplay] Extracted images:", images.length, "text:", text?.substring?.(0, 100));
  
  return (
    <div>
      {/* Show extracted images */}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {images.map((src, i) => (
            <img
              key={i}
              src={src}
              alt={`Tool output ${i + 1}`}
              className="max-w-[200px] max-h-[150px] rounded-lg border border-gray-600 object-contain bg-gray-900"
              onError={(e) => {
                e.target.style.display = "none";
              }}
            />
          ))}
        </div>
      )}
      
      {/* Show text output */}
      <pre className="mt-1 text-xs text-gray-300 bg-gray-800/50 rounded p-2 overflow-x-auto max-h-32 overflow-y-auto">
        {text}
      </pre>
    </div>
  );
}

/**
 * Get icon for a modality type.
 */
function getModalityIcon(modality) {
  if (modality.includes("image")) return Image;
  if (modality.includes("video")) return Video;
  if (modality.includes("text")) return FileText;
  return Zap;
}

/**
 * Chat view component with agent info sidebar and multimodal input.
 */
export default function ChatView({
  agent,
  messages,
  isLoading,
  systemPrompt,
  onSystemPromptChange,
  onSendMessage,
  onMediaUpload,
  onNewChat,
}) {
  const [showAgentInfo, setShowAgentInfo] = useState(true);
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Parse supported modalities
  const inputModes = agent.defaultInputModes || agent.default_input_modes || [];
  const supportsImages = inputModes.some(
    (m) => m.includes("image") || m === "image/png" || m === "image/jpeg"
  );
  const supportsVideo = inputModes.some(
    (m) => m.includes("video") || m === "video/mp4"
  );

  return (
    <div className="flex h-full">
      {/* Agent Info Sidebar */}
      {showAgentInfo && (
        <AgentInfoSidebar
          agent={agent}
          inputModes={inputModes}
          systemPrompt={systemPrompt}
          showSystemPrompt={showSystemPrompt}
          onSystemPromptChange={onSystemPromptChange}
          onToggleSystemPrompt={() => setShowSystemPrompt(!showSystemPrompt)}
          onClose={() => setShowAgentInfo(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 bg-gray-800/50">
          <div className="flex items-center gap-3">
            {!showAgentInfo && (
              <button
                onClick={() => setShowAgentInfo(true)}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                title="Show agent info"
              >
                <Bot className="w-5 h-5" />
              </button>
            )}
            <div>
              <h3 className="font-semibold">{agent.name}</h3>
              <p className="text-xs text-gray-400">
                {messages.length} messages
              </p>
            </div>
          </div>
          <button
            onClick={onNewChat}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <EmptyState agent={agent} inputModes={inputModes} />
          ) : (
            <GroupedMessages messages={messages} isLoading={isLoading} />
          )}
          {isLoading && (
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Agent is thinking...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <ChatInput
          onSendMessage={onSendMessage}
          onMediaUpload={onMediaUpload}
          isLoading={isLoading}
          supportsImages={supportsImages}
          supportsVideo={supportsVideo}
        />
      </div>
    </div>
  );
}

/**
 * Agent info sidebar showing skills and modalities.
 */
function AgentInfoSidebar({
  agent,
  inputModes,
  systemPrompt,
  showSystemPrompt,
  onSystemPromptChange,
  onToggleSystemPrompt,
  onClose,
}) {
  return (
    <div className="w-72 border-r border-gray-700 bg-gray-800/50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <h4 className="font-semibold text-sm">Agent Info</h4>
        <button
          onClick={onClose}
          className="p-1 text-gray-400 hover:text-white rounded transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Agent Details */}
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-blue-600/20 rounded-lg">
              <Bot className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h5 className="font-medium">{agent.name}</h5>
              {agent.version && (
                <span className="text-xs text-gray-500">v{agent.version}</span>
              )}
            </div>
          </div>
          <p className="text-sm text-gray-400">{agent.description}</p>
        </div>

        {/* Supported Modalities */}
        {inputModes.length > 0 && (
          <div>
            <h6 className="text-xs font-medium text-gray-500 uppercase mb-2">
              Supported Input
            </h6>
            <div className="flex flex-wrap gap-2">
              {inputModes.map((mode) => {
                const Icon = getModalityIcon(mode);
                return (
                  <span
                    key={mode}
                    className="flex items-center gap-1 text-xs px-2 py-1 bg-blue-600/20 text-blue-400 rounded"
                  >
                    <Icon className="w-3 h-3" />
                    {mode}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* Capabilities */}
        {agent.capabilities && (
          <div>
            <h6 className="text-xs font-medium text-gray-500 uppercase mb-2">
              Capabilities
            </h6>
            <div className="flex flex-wrap gap-2">
              {agent.capabilities.streaming && (
                <span className="text-xs px-2 py-1 bg-purple-600/20 text-purple-400 rounded">
                  Streaming
                </span>
              )}
              {agent.capabilities.pushNotifications && (
                <span className="text-xs px-2 py-1 bg-purple-600/20 text-purple-400 rounded">
                  Push Notifications
                </span>
              )}
            </div>
          </div>
        )}

        {/* Skills */}
        {agent.skills && agent.skills.length > 0 && (
          <div>
            <h6 className="text-xs font-medium text-gray-500 uppercase mb-2">
              Skills
            </h6>
            <div className="space-y-2">
              {agent.skills.map((skill) => (
                <div
                  key={skill.id}
                  className="p-2 bg-gray-700/50 rounded-lg"
                >
                  <h6 className="text-sm font-medium">{skill.name}</h6>
                  <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                    {skill.description}
                  </p>
                  {skill.tags && skill.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {skill.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="text-[10px] px-1.5 py-0.5 bg-gray-600 text-gray-300 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* System Prompt */}
        <div>
          <button
            onClick={onToggleSystemPrompt}
            className="flex items-center justify-between w-full text-xs font-medium text-gray-500 uppercase mb-2"
          >
            <span className="flex items-center gap-1">
              <Settings className="w-3 h-3" />
              System Prompt
            </span>
            {showSystemPrompt ? (
              <ChevronUp className="w-3 h-3" />
            ) : (
              <ChevronDown className="w-3 h-3" />
            )}
          </button>
          {showSystemPrompt && (
            <textarea
              value={systemPrompt}
              onChange={(e) => onSystemPromptChange(e.target.value)}
              placeholder="Optional system prompt to customize agent behavior..."
              rows={4}
              className="w-full px-3 py-2 text-sm bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500 resize-none"
            />
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Empty state when no messages.
 */
function EmptyState({ agent, inputModes }) {
  const supportsImages = inputModes.some((m) => m.includes("image"));
  const supportsVideo = inputModes.some((m) => m.includes("video"));

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="p-4 bg-blue-600/20 rounded-2xl mb-4">
        <Bot className="w-12 h-12 text-blue-400" />
      </div>
      <h3 className="text-xl font-semibold mb-2">Chat with {agent.name}</h3>
      <p className="text-gray-400 max-w-md mb-6">{agent.description}</p>

      {/* Quick tips */}
      <div className="flex flex-wrap justify-center gap-2">
        <span className="flex items-center gap-1 text-xs px-3 py-1.5 bg-gray-800 text-gray-300 rounded-full">
          <FileText className="w-3 h-3" />
          Send text messages
        </span>
        {supportsImages && (
          <span className="flex items-center gap-1 text-xs px-3 py-1.5 bg-gray-800 text-gray-300 rounded-full">
            <Image className="w-3 h-3" />
            Upload images
          </span>
        )}
        {supportsVideo && (
          <span className="flex items-center gap-1 text-xs px-3 py-1.5 bg-gray-800 text-gray-300 rounded-full">
            <Video className="w-3 h-3" />
            Upload videos
          </span>
        )}
        <span className="flex items-center gap-1 text-xs px-3 py-1.5 bg-gray-800 text-gray-300 rounded-full">
          <FileText className="w-3 h-3 text-amber-400" />
          Upload documents (PDF, Word, CSV…)
        </span>
      </div>

      {/* Example prompts */}
      {agent.skills && agent.skills.length > 0 && agent.skills[0].examples && (
        <div className="mt-6">
          <p className="text-xs text-gray-500 mb-2">Try asking:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {agent.skills[0].examples.slice(0, 3).map((example, i) => (
              <span
                key={i}
                className="text-xs px-3 py-1.5 bg-gray-700 text-gray-300 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors"
              >
                "{example}"
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Renders the message list, grouping thoughts and tools into separate
 * collapsible blocks.
 */
function GroupedMessages({ messages, isLoading }) {
  const groups = useMemo(() => groupMessages(messages), [messages]);

  return (
    <>
      {groups.map((group) => {
        if (group.kind === "thought") {
          return (
            <ThoughtGroup
              key={group.id}
              messages={group.messages}
              isLoading={isLoading}
            />
          );
        }
        if (group.kind === "tools") {
          return (
            <ToolsGroup
              key={group.id}
              messages={group.messages}
              isLoading={isLoading}
            />
          );
        }
        return (
          <MessageBubble key={group.message.id} message={group.message} />
        );
      })}
    </>
  );
}

/**
 * Collapsible "Thought" block — shows the LLM's reasoning text.
 */
function ThoughtGroup({ messages, isLoading }) {
  const [expanded, setExpanded] = useState(false);

  const isInProgress = isLoading && messages.length > 0;
  const summary = isInProgress ? "Thinking…" : "Thought";

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] w-full">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-2 group w-full text-left"
        >
          {isInProgress ? (
            <Loader2 className="w-4 h-4 text-purple-400 animate-spin flex-shrink-0" />
          ) : (
            <Brain className="w-4 h-4 text-purple-400 flex-shrink-0" />
          )}
          <span className="text-sm text-gray-400 group-hover:text-gray-200 transition-colors">
            {summary}
          </span>
          <ChevronRight
            className={`w-3.5 h-3.5 text-gray-500 transition-transform duration-200 ${
              expanded ? "rotate-90" : ""
            }`}
          />
        </button>

        {expanded && (
          <div className="mt-2 ml-6 border-l-2 border-purple-800/40 pl-4 space-y-2">
            {messages.map((msg) => (
              <div key={msg.id} className="text-sm text-gray-300 prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Collapsible "Tools" block — shows tool calls and results.
 */
function ToolsGroup({ messages, isLoading }) {
  const [expanded, setExpanded] = useState(false);

  const toolCalls = messages.filter(
    (m) => m.type === "tool_call" || m.type === "tool_start"
  );
  const toolResults = messages.filter(
    (m) => m.type === "tool_result" || m.type === "tool_end"
  );

  const isInProgress = isLoading && toolCalls.length > toolResults.length;
  const toolNames = [...new Set(toolCalls.map((m) => m.toolName).filter(Boolean))];

  let summary;
  if (isInProgress) {
    const currentTool = toolCalls[toolCalls.length - 1]?.toolName;
    summary = currentTool ? `Using ${currentTool}…` : "Running tools…";
  } else if (toolNames.length === 1) {
    summary = `Used ${toolNames[0]}`;
  } else {
    summary = `Used ${toolCalls.length} tool${toolCalls.length !== 1 ? "s" : ""}`;
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] w-full">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-2 group w-full text-left"
        >
          {isInProgress ? (
            <Loader2 className="w-4 h-4 text-blue-400 animate-spin flex-shrink-0" />
          ) : (
            <Sparkles className="w-4 h-4 text-amber-400 flex-shrink-0" />
          )}
          <span className="text-sm text-gray-400 group-hover:text-gray-200 transition-colors">
            {summary}
          </span>
          <ChevronRight
            className={`w-3.5 h-3.5 text-gray-500 transition-transform duration-200 ${
              expanded ? "rotate-90" : ""
            }`}
          />
        </button>

        {expanded && (
          <div className="mt-2 ml-6 border-l-2 border-gray-700 pl-4 space-y-2">
            {messages.map((msg) => (
              <ThinkingStep key={msg.id} message={msg} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * A single step inside the expanded thinking group.
 */
function ThinkingStep({ message }) {
  const [showDetails, setShowDetails] = useState(false);
  const isToolCall = message.type === "tool_call" || message.type === "tool_start";
  const isToolResult = message.type === "tool_result" || message.type === "tool_end";
  const isThought = message.type === "llm_thought";

  if (isThought) {
    return (
      <div className="text-sm">
        <button
          onClick={() => setShowDetails((v) => !v)}
          className="flex items-center gap-1.5 text-purple-400 hover:text-purple-300 transition-colors"
        >
          <Brain className="w-3.5 h-3.5" />
          <span className="font-medium">Thought</span>
          {message.content && (
            <ChevronRight
              className={`w-3 h-3 transition-transform duration-200 ${
                showDetails ? "rotate-90" : ""
              }`}
            />
          )}
        </button>
        {showDetails && message.content && (
          <div className="mt-1 ml-5 text-xs text-gray-400 prose prose-invert prose-xs max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
    );
  }

  if (isToolCall) {
    return (
      <div className="text-sm">
        <button
          onClick={() => setShowDetails((v) => !v)}
          className="flex items-center gap-1.5 text-amber-400 hover:text-amber-300 transition-colors"
        >
          <Play className="w-3.5 h-3.5" />
          <span className="font-medium">{message.toolName || "Tool"}</span>
          {message.toolInput && (
            <ChevronRight
              className={`w-3 h-3 transition-transform duration-200 ${
                showDetails ? "rotate-90" : ""
              }`}
            />
          )}
        </button>
        {showDetails && message.toolInput && (
          <div className="mt-1 ml-5">
            <span className="text-[10px] text-gray-500 uppercase">Input:</span>
            <pre className="mt-0.5 text-xs text-gray-300 bg-gray-800/50 rounded p-2 overflow-x-auto max-h-24 overflow-y-auto">
              {message.toolInput}
            </pre>
          </div>
        )}
      </div>
    );
  }

  if (isToolResult) {
    // Extract images from tool output for inline display
    const hasOutput = Boolean(message.toolOutput);
    const { images } = hasOutput
      ? extractImagesFromOutput(message.toolOutput)
      : { images: [] };

    return (
      <div className="text-sm">
        <button
          onClick={() => setShowDetails((v) => !v)}
          className="flex items-center gap-1.5 text-green-400 hover:text-green-300 transition-colors"
        >
          <CheckCircle className="w-3.5 h-3.5" />
          <span className="font-medium">{message.toolName || "Result"}</span>
          {hasOutput && (
            <ChevronRight
              className={`w-3 h-3 transition-transform duration-200 ${
                showDetails ? "rotate-90" : ""
              }`}
            />
          )}
        </button>

        {/* Always show extracted images outside the collapsible */}
        {images.length > 0 && (
          <div className="mt-1 ml-5 flex flex-wrap gap-2">
            {images.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={`Tool output ${i + 1}`}
                className="max-w-[250px] max-h-[160px] rounded-lg border border-gray-600 object-contain bg-gray-900"
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />
            ))}
          </div>
        )}

        {showDetails && hasOutput && (
          <div className="mt-1 ml-5">
            <span className="text-[10px] text-gray-500 uppercase">Output:</span>
            <ToolOutputDisplay output={message.toolOutput} />
          </div>
        )}
      </div>
    );
  }

  // Fallback — shouldn't happen
  return null;
}

/**
 * Individual message bubble.
 */
function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isError = message.type === "error";
  const isStreaming = message.type === "streaming";

  // Thinking-type messages that somehow weren't grouped (edge case / fallback)
  if (isThoughtType(message.type)) {
    return <ThoughtGroup messages={[message]} isLoading={false} />;
  }
  if (isToolType(message.type)) {
    return <ToolsGroup messages={[message]} isLoading={false} />;
  }

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white"
            : isError
            ? "bg-red-600/20 text-red-400 border border-red-600/30"
            : isSystem
            ? "bg-yellow-600/20 text-yellow-400 border border-yellow-600/30"
            : isStreaming
            ? "bg-gray-800 text-gray-100 border border-blue-500/30"
            : "bg-gray-800 text-gray-100"
        }`}
      >
        {/* Media preview for user messages */}
        {message.media && message.media.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {message.media.map((m, i) => (
              <div key={i} className="relative">
                {m.type.startsWith("image/") ? (
                  <img
                    src={m.url}
                    alt={m.name}
                    className="max-w-[200px] max-h-[150px] rounded-lg object-cover"
                  />
                ) : m.type.startsWith("video/") ? (
                  <video
                    src={m.url}
                    className="max-w-[200px] max-h-[150px] rounded-lg"
                    controls
                  />
                ) : (
                  <div className="flex items-center gap-2 px-3 py-2 bg-gray-700 rounded-lg">
                    <Paperclip className="w-4 h-4" />
                    <span className="text-sm">{m.name}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Message content */}
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : isStreaming ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
            <span className="inline-block w-2 h-4 ml-0.5 bg-blue-400 animate-pulse rounded-sm" />
          </div>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Timestamp */}
        {message.timestamp && !isStreaming && (
          <p
            className={`text-[10px] mt-1 ${
              isUser ? "text-blue-200" : "text-gray-500"
            }`}
          >
            {new Date(message.timestamp).toLocaleTimeString()}
          </p>
        )}
      </div>
    </div>
  );
}

const DOCUMENT_MIMES = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/msword",
  "text/plain",
  "text/markdown",
  "text/csv",
]);

const DOCUMENT_EXTS = new Set([".pdf", ".docx", ".doc", ".txt", ".md", ".csv"]);

function isDocumentFile(file) {
  if (DOCUMENT_MIMES.has(file.type)) return true;
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  return DOCUMENT_EXTS.has(ext);
}

/**
 * Chat input with multimodal support.
 */
function ChatInput({
  onSendMessage,
  onMediaUpload,
  isLoading,
  supportsImages,
  supportsVideo,
}) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        150
      )}px`;
    }
  }, [text]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isLoading) return;

    if (files.length > 0) {
      onMediaUpload(files, text.trim() || undefined);
      setFiles([]);
      setText("");
    } else if (text.trim()) {
      onSendMessage(text.trim());
      setText("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles((prev) => [...prev, ...selectedFiles]);
    e.target.value = "";
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Build accept string — MIME types AND extensions for broad browser compatibility
  const acceptTypes = [
    "image/*",
    "video/*",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/markdown",
    "text/csv",
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".csv",
  ];

  return (
    <div className="border-t border-gray-700 bg-gray-800/50 p-4">
      {/* File previews */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {files.map((file, i) => (
            <div
              key={i}
              className="relative group flex items-center gap-2 px-3 py-2 bg-gray-700 rounded-lg"
            >
              {file.type.startsWith("image/") ? (
                <Image className="w-4 h-4 text-blue-400" />
              ) : file.type.startsWith("video/") ? (
                <Video className="w-4 h-4 text-purple-400" />
              ) : isDocumentFile(file) ? (
                <FileText className="w-4 h-4 text-amber-400" />
              ) : (
                <Paperclip className="w-4 h-4 text-gray-400" />
              )}
              <span className="text-sm text-gray-300 max-w-[150px] truncate">
                {file.name}
              </span>
              <button
                onClick={() => removeFile(i)}
                className="p-0.5 text-gray-400 hover:text-red-400 transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        {/* File upload button — always visible */}
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptTypes.join(",")}
          multiple
          onChange={handleFileSelect}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className="p-3 text-gray-400 hover:text-white hover:bg-gray-700 rounded-xl transition-colors disabled:opacity-50"
          title="Attach files (images, video, PDF, Word, text)"
        >
          <Paperclip className="w-5 h-5" />
        </button>

        {/* Text input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              files.length > 0
                ? "Add a message (optional)..."
                : "Type a message..."
            }
            rows={1}
            disabled={isLoading}
            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl focus:outline-none focus:border-blue-500 resize-none disabled:opacity-50"
          />
        </div>

        {/* Send button */}
        <button
          type="submit"
          disabled={isLoading || (!text.trim() && files.length === 0)}
          className="p-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-xl transition-colors"
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>

      {/* Supported modalities hint */}
      <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
        <span>Supports:</span>
        <span className="flex items-center gap-1">
          <FileText className="w-3 h-3" />
          Text
        </span>
        {supportsImages && (
          <span className="flex items-center gap-1">
            <Image className="w-3 h-3" />
            Images
          </span>
        )}
        {supportsVideo && (
          <span className="flex items-center gap-1">
            <Video className="w-3 h-3" />
            Video
          </span>
        )}
        <span className="flex items-center gap-1">
          <FileText className="w-3 h-3 text-amber-500" />
          Documents
        </span>
      </div>
    </div>
  );
}
