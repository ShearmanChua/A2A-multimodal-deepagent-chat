import { useState, useCallback, useEffect } from "react";
import * as api from "./api";
import Header from "./components/Header";
import AgentSelector from "./components/AgentSelector";
import AgentRegistry from "./components/AgentRegistry";
import ChatView from "./components/ChatView";

/**
 * Main App component with three main views:
 * 1. Agent Selection - Choose an agent to chat with
 * 2. Agent Registration - Register new A2A agents
 * 3. Chat - Chat interface with the selected agent
 */
export default function App() {
  // Tab state: "agents" | "register" | "chat"
  const [activeTab, setActiveTab] = useState("agents");
  
  // Agent registry state
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [loadingAgents, setLoadingAgents] = useState(true);
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [useStreaming, setUseStreaming] = useState(false); // Disable streaming by default

  // Fetch registered agents on mount
  useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    setLoadingAgents(true);
    try {
      const data = await api.listAgents();
      setAgents(data.agents || []);
    } catch (err) {
      console.error("Failed to fetch agents:", err);
    } finally {
      setLoadingAgents(false);
    }
  };

  // Handle agent registration
  const handleRegisterAgent = useCallback(async (agentData) => {
    try {
      const result = await api.registerAgent(agentData);
      setAgents((prev) => [...prev, result.agent]);
      return result;
    } catch (err) {
      throw err;
    }
  }, []);

  // Handle agent removal
  const handleRemoveAgent = useCallback(async (agentId) => {
    try {
      await api.removeAgent(agentId);
      setAgents((prev) => prev.filter((a) => a.id !== agentId));
      if (selectedAgent?.id === agentId) {
        setSelectedAgent(null);
        setActiveTab("agents");
      }
    } catch (err) {
      console.error("Failed to remove agent:", err);
    }
  }, [selectedAgent]);

  // Handle agent refresh
  const handleRefreshAgent = useCallback(async (agentId) => {
    try {
      const result = await api.refreshAgent(agentId);
      setAgents((prev) =>
        prev.map((a) => (a.id === agentId ? result.agent : a))
      );
      if (selectedAgent?.id === agentId) {
        setSelectedAgent(result.agent);
      }
    } catch (err) {
      console.error("Failed to refresh agent:", err);
    }
  }, [selectedAgent]);

  // Handle agent selection and start chat
  const handleSelectAgent = useCallback(async (agent) => {
    setSelectedAgent(agent);
    setMessages([]);
    setSessionId(null);
    setActiveTab("chat");
    
    // Create a new session for this agent
    try {
      const result = await api.createSession(agent.id, systemPrompt);
      setSessionId(result.session.sessionId);
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  }, [systemPrompt]);

  // Handle sending a message with streaming for tool call display
  // and token-by-token LLM output
  const handleSendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading || !selectedAgent) return;

    // Ensure we have a session
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      try {
        const result = await api.createSession(selectedAgent.id, systemPrompt);
        currentSessionId = result.session.sessionId;
        setSessionId(currentSessionId);
      } catch (err) {
        const errorMessage = {
          id: Date.now(),
          role: "system",
          content: `Error creating session: ${err.message}`,
          type: "error",
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errorMessage]);
        return;
      }
    }

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: "user",
      content: text,
      type: "text",
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      let finalContent = "";

      console.log("[App] About to call sendStreamingMessage");
      console.log("[App] currentSessionId:", currentSessionId);
      
      if (!currentSessionId) {
        console.error("[App] ERROR: currentSessionId is null/undefined after session creation!");
        throw new Error("Session ID is missing");
      }

      // Mutable streaming state for accumulating tokens
      let streamingMsgId = Date.now() + 0.5;
      let streamingMsgCreated = false;
      let accumulatedTokens = "";
      let totalAccumulatedTokens = "";

      try {
        console.log("[App] Calling api.sendStreamingMessage...");
        await api.sendStreamingMessage(currentSessionId, text, (event) => {
          console.log("[App] Streaming event:", event.type);
          
          if (event.type === "tool_call" || event.type === "tool_start") {
            const msgId = Date.now() + Math.random();
            const toolMsg = {
              id: msgId,
              role: "assistant",
              content: `🔧 Calling: ${event.toolName || "tool"}`,
              type: "tool_call",
              toolName: event.toolName,
              toolInput: event.toolInput,
              timestamp: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, toolMsg]);
          } else if (event.type === "tool_result" || event.type === "tool_end") {
            const msgId = Date.now() + Math.random();
            const toolMsg = {
              id: msgId,
              role: "assistant",
              content: `✅ Completed: ${event.toolName || "tool"}`,
              type: "tool_result",
              toolName: event.toolName,
              toolOutput: event.toolOutput,
              timestamp: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, toolMsg]);
          } else if (event.type === "llm_thought") {
            // LLM reasoning / thoughts — the full thought text arrives
            // as a single event (not streamed token-by-token).
            if (event.content) {
              const msgId = Date.now() + Math.random();
              const thoughtMsg = {
                id: msgId,
                role: "assistant",
                content: event.content,
                type: "llm_thought",
                toolCalls: event.toolCalls || [],
                timestamp: new Date().toISOString(),
              };
              setMessages((prev) => [...prev, thoughtMsg]);
            }
          } else if (event.type === "token") {
            // Token-by-token streaming — create or update the
            // streaming assistant message in real time
            console.log("[App] Token event received:", event.content?.slice(0, 50), "streamingMsgId:", streamingMsgId);
            if (event.content) {
              accumulatedTokens += event.content;
              totalAccumulatedTokens += event.content;
              streamingMsgCreated = true;

              // Always use functional update to handle race conditions
              const currentId = streamingMsgId;
              const currentContent = accumulatedTokens;
              console.log("[App] Updating streaming message, id:", currentId, "content length:", currentContent.length);
              setMessages((prev) => {
                // Check if the streaming message already exists
                const existingIndex = prev.findIndex((m) => m.id === currentId);
                console.log("[App] setMessages callback - existingIndex:", existingIndex, "prev.length:", prev.length);
                if (existingIndex >= 0) {
                  // Update the existing streaming message
                  return prev.map((m) =>
                    m.id === currentId
                      ? { ...m, content: currentContent }
                      : m
                  );
                } else {
                  // Create a new streaming message
                  console.log("[App] Creating new streaming message with id:", currentId);
                  return [
                    ...prev,
                    {
                      id: currentId,
                      role: "assistant",
                      content: currentContent,
                      type: "streaming",
                      timestamp: new Date().toISOString(),
                    },
                  ];
                }
              });
            }
          } else if (event.type === "final_response" || event.type === "final") {
            // Final event — replace the current streaming message with
            // the complete response, or create a new message if no
            // streaming bubble exists (e.g. after token_done reset).
            finalContent = event.content || accumulatedTokens || totalAccumulatedTokens;
            if (streamingMsgCreated) {
              const currentId = streamingMsgId;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === currentId
                    ? { ...m, content: finalContent, type: "text" }
                    : m
                )
              );
            } else if (finalContent) {
              // No active streaming bubble — create a new finalised message
              const newMsgId = Date.now() + Math.random();
              setMessages((prev) => [
                ...prev,
                {
                  id: newMsgId,
                  role: "assistant",
                  content: finalContent,
                  type: "text",
                  timestamp: new Date().toISOString(),
                },
              ]);
            }
          } else if (event.type === "done") {
            // Stream complete — finalise any remaining streaming bubble
            if (!finalContent && totalAccumulatedTokens) {
              finalContent = totalAccumulatedTokens;
            }
            if (streamingMsgCreated && accumulatedTokens) {
              const currentId = streamingMsgId;
              const currentContent = finalContent || accumulatedTokens;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === currentId
                    ? { ...m, content: currentContent, type: "text" }
                    : m
                )
              );
            } else if (streamingMsgCreated && !accumulatedTokens) {
              // Remove empty streaming message
              const currentId = streamingMsgId;
              setMessages((prev) => prev.filter((m) => m.id !== currentId));
            } else if (!streamingMsgCreated && finalContent) {
              // Edge case: no streaming bubble was ever created but we
              // have final content — create a message now
              setMessages((prev) => {
                // Only add if not already present from the "final" handler
                const alreadyHasFinal = prev.some(
                  (m) => m.role === "assistant" && m.content === finalContent && m.type === "text"
                );
                if (alreadyHasFinal) return prev;
                return [
                  ...prev,
                  {
                    id: Date.now() + Math.random(),
                    role: "assistant",
                    content: finalContent,
                    type: "text",
                    timestamp: new Date().toISOString(),
                  },
                ];
              });
            }
            console.log("[App] Stream done, final content length:", finalContent.length);
          } else if (event.type === "error") {
            throw new Error(event.error || "Streaming error");
          }
        });

        // If we got tokens but the streaming message wasn't finalised
        // (e.g. no "done" event), add the final content
        if (!streamingMsgCreated && totalAccumulatedTokens && !finalContent) {
          finalContent = totalAccumulatedTokens;
          const assistantMessage = {
            id: Date.now() + 1,
            role: "assistant",
            content: finalContent,
            type: "text",
            timestamp: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, assistantMessage]);
        }
      } catch (streamErr) {
        // Streaming failed, fall back to non-streaming
        console.error("[App] Streaming failed, falling back to non-streaming:", streamErr.message);
        console.error("[App] Full error:", streamErr);
        
        const result = await api.sendMessage(currentSessionId, text);
        
        const assistantMessage = {
          id: Date.now() + 1,
          role: "assistant",
          content: result.response.text,
          type: result.response.type || "text",
          state: result.response.state,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }

    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        role: "system",
        content: `Error: ${err.message}`,
        type: "error",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [selectedAgent, sessionId, systemPrompt, isLoading]);

  // Handle media upload with streaming for tool call display
  // and token-by-token LLM output
  const handleMediaUpload = useCallback(async (files, query) => {
    if (files.length === 0 || isLoading || !selectedAgent) return;

    // Ensure we have a session
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      try {
        const result = await api.createSession(selectedAgent.id, systemPrompt);
        currentSessionId = result.session.sessionId;
        setSessionId(currentSessionId);
      } catch (err) {
        const errorMessage = {
          id: Date.now(),
          role: "system",
          content: `Error creating session: ${err.message}`,
          type: "error",
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errorMessage]);
        return;
      }
    }

    // Add user message with media preview
    const mediaPreview = Array.from(files).map((f) => ({
      name: f.name,
      type: f.type,
      url: URL.createObjectURL(f),
    }));

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: query || "Analyze the uploaded media.",
      type: "media",
      media: mediaPreview,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      let finalContent = "";

      console.log("[App] About to call uploadMediaStreaming");
      console.log("[App] currentSessionId:", currentSessionId);

      let streamingMsgId = Date.now() + 0.5;
      let streamingMsgCreated = false;
      let accumulatedTokens = "";
      let totalAccumulatedTokens = "";

      try {
        console.log("[App] Calling api.uploadMediaStreaming...");
        await api.uploadMediaStreaming(currentSessionId, files, query, (event) => {
          console.log("[App] Streaming upload event:", event.type);
          
          if (event.type === "tool_call" || event.type === "tool_start") {
            const msgId = Date.now() + Math.random();
            const toolMsg = {
              id: msgId,
              role: "assistant",
              content: `🔧 Calling: ${event.toolName || "tool"}`,
              type: "tool_call",
              toolName: event.toolName,
              toolInput: event.toolInput,
              timestamp: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, toolMsg]);
          } else if (event.type === "tool_result" || event.type === "tool_end") {
            const msgId = Date.now() + Math.random();
            const toolMsg = {
              id: msgId,
              role: "assistant",
              content: `✅ Completed: ${event.toolName || "tool"}`,
              type: "tool_result",
              toolName: event.toolName,
              toolOutput: event.toolOutput,
              timestamp: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, toolMsg]);
          } else if (event.type === "llm_thought") {
            // LLM reasoning / thoughts — arrives as a single event.
            if (event.content) {
              const msgId = Date.now() + Math.random();
              const thoughtMsg = {
                id: msgId,
                role: "assistant",
                content: event.content,
                type: "llm_thought",
                toolCalls: event.toolCalls || [],
                timestamp: new Date().toISOString(),
              };
              setMessages((prev) => [...prev, thoughtMsg]);
            }
          } else if (event.type === "token") {
            // Token-by-token streaming — create or update the
            // streaming assistant message in real time
            if (event.content) {
              accumulatedTokens += event.content;
              totalAccumulatedTokens += event.content;
              streamingMsgCreated = true;

              // Always use functional update to handle race conditions
              const currentId = streamingMsgId;
              const currentContent = accumulatedTokens;
              setMessages((prev) => {
                // Check if the streaming message already exists
                const existingIndex = prev.findIndex((m) => m.id === currentId);
                if (existingIndex >= 0) {
                  // Update the existing streaming message
                  return prev.map((m) =>
                    m.id === currentId
                      ? { ...m, content: currentContent }
                      : m
                  );
                } else {
                  // Create a new streaming message
                  return [
                    ...prev,
                    {
                      id: currentId,
                      role: "assistant",
                      content: currentContent,
                      type: "streaming",
                      timestamp: new Date().toISOString(),
                    },
                  ];
                }
              });
            }
          } else if (event.type === "final_response" || event.type === "final") {
            finalContent = event.content || accumulatedTokens || totalAccumulatedTokens;
            if (streamingMsgCreated) {
              const currentId = streamingMsgId;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === currentId
                    ? { ...m, content: finalContent, type: "media_analysis" }
                    : m
                )
              );
            } else if (finalContent) {
              // No active streaming bubble — create a new finalised message
              const newMsgId = Date.now() + Math.random();
              setMessages((prev) => [
                ...prev,
                {
                  id: newMsgId,
                  role: "assistant",
                  content: finalContent,
                  type: "media_analysis",
                  timestamp: new Date().toISOString(),
                },
              ]);
            }
          } else if (event.type === "done") {
            if (!finalContent && totalAccumulatedTokens) {
              finalContent = totalAccumulatedTokens;
            }
            if (streamingMsgCreated && accumulatedTokens) {
              const currentId = streamingMsgId;
              const currentContent = finalContent || accumulatedTokens;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === currentId
                    ? { ...m, content: currentContent, type: "media_analysis" }
                    : m
                )
              );
            } else if (streamingMsgCreated && !accumulatedTokens) {
              const currentId = streamingMsgId;
              setMessages((prev) => prev.filter((m) => m.id !== currentId));
            } else if (!streamingMsgCreated && finalContent) {
              // Edge case: no streaming bubble but we have final content
              setMessages((prev) => {
                const alreadyHasFinal = prev.some(
                  (m) => m.role === "assistant" && m.content === finalContent && m.type === "media_analysis"
                );
                if (alreadyHasFinal) return prev;
                return [
                  ...prev,
                  {
                    id: Date.now() + Math.random(),
                    role: "assistant",
                    content: finalContent,
                    type: "media_analysis",
                    timestamp: new Date().toISOString(),
                  },
                ];
              });
            }
            console.log("[App] Stream done, final content length:", finalContent.length);
          } else if (event.type === "error") {
            throw new Error(event.error || "Streaming error");
          }
        });

        // If we got tokens but the streaming message wasn't finalised
        if (!streamingMsgCreated && totalAccumulatedTokens && !finalContent) {
          finalContent = totalAccumulatedTokens;
          const assistantMessage = {
            id: Date.now() + 1,
            role: "assistant",
            content: finalContent,
            type: "media_analysis",
            timestamp: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, assistantMessage]);
        }
      } catch (streamErr) {
        // Streaming failed, fall back to non-streaming
        console.error("[App] Streaming upload failed, falling back to non-streaming:", streamErr.message);
        console.error("[App] Full error:", streamErr);
        
        const result = await api.uploadMedia(currentSessionId, files, query);

        const assistantMessage = {
          id: Date.now() + 1,
          role: "assistant",
          content: result.response.text,
          type: "media_analysis",
          state: result.response.state,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }

    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        role: "system",
        content: `Error: ${err.message}`,
        type: "error",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [selectedAgent, sessionId, systemPrompt, isLoading]);

  // Handle new chat (clear messages and create new session)
  const handleNewChat = useCallback(async () => {
    setMessages([]);
    setSessionId(null);
    
    // Create a new session if we have a selected agent
    if (selectedAgent) {
      try {
        const result = await api.createSession(selectedAgent.id, systemPrompt);
        setSessionId(result.session.sessionId);
      } catch (err) {
        console.error("Failed to create new session:", err);
      }
    }
  }, [selectedAgent, systemPrompt]);

  // Handle back to agent selection
  const handleBackToAgents = useCallback(() => {
    setActiveTab("agents");
  }, []);

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
      {/* Header with tabs */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        selectedAgent={selectedAgent}
        onBackToAgents={handleBackToAgents}
      />

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        {activeTab === "agents" && (
          <AgentSelector
            agents={agents}
            loading={loadingAgents}
            onSelectAgent={handleSelectAgent}
            onRefreshAgent={handleRefreshAgent}
            onRemoveAgent={handleRemoveAgent}
            onRefreshList={fetchAgents}
          />
        )}

        {activeTab === "register" && (
          <AgentRegistry
            onRegister={handleRegisterAgent}
            onSuccess={() => {
              fetchAgents();
              setActiveTab("agents");
            }}
          />
        )}

        {activeTab === "chat" && selectedAgent && (
          <ChatView
            agent={selectedAgent}
            messages={messages}
            isLoading={isLoading}
            systemPrompt={systemPrompt}
            onSystemPromptChange={setSystemPrompt}
            onSendMessage={handleSendMessage}
            onMediaUpload={handleMediaUpload}
            onNewChat={handleNewChat}
          />
        )}

        {activeTab === "chat" && !selectedAgent && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-gray-400 mb-4">No agent selected</p>
              <button
                onClick={() => setActiveTab("agents")}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
              >
                Select an Agent
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
