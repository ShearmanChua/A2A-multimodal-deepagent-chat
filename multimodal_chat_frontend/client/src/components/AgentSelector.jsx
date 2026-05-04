import { useState } from "react";
import {
  Bot,
  RefreshCw,
  Trash2,
  MessageSquare,
  CheckCircle,
  XCircle,
  Loader2,
  ExternalLink,
  Image,
  Video,
  FileText,
  Zap,
} from "lucide-react";

/**
 * Agent selection tab - displays registered agents and allows selection.
 */
export default function AgentSelector({
  agents,
  loading,
  onSelectAgent,
  onRefreshAgent,
  onRemoveAgent,
  onRefreshList,
}) {
  const [refreshingId, setRefreshingId] = useState(null);
  const [removingId, setRemovingId] = useState(null);

  const handleRefresh = async (agentId) => {
    setRefreshingId(agentId);
    try {
      await onRefreshAgent(agentId);
    } finally {
      setRefreshingId(null);
    }
  };

  const handleRemove = async (agentId) => {
    if (!confirm("Are you sure you want to remove this agent?")) return;
    setRemovingId(agentId);
    try {
      await onRemoveAgent(agentId);
    } finally {
      setRemovingId(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex items-center gap-3 text-gray-400">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Loading agents...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">Registered Agents</h2>
            <p className="text-gray-400 mt-1">
              Select an agent to start chatting
            </p>
          </div>
          <button
            onClick={onRefreshList}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>

        {/* Agent list */}
        {agents.length === 0 ? (
          <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
            <Bot className="w-16 h-16 mx-auto text-gray-600 mb-4" />
            <h3 className="text-xl font-semibold text-gray-300 mb-2">
              No Agents Registered
            </h3>
            <p className="text-gray-500 mb-4">
              Register an A2A agent to get started
            </p>
          </div>
        ) : (
          <div className="grid gap-4">
            {agents.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                onSelect={() => onSelectAgent(agent)}
                onRefresh={() => handleRefresh(agent.id)}
                onRemove={() => handleRemove(agent.id)}
                isRefreshing={refreshingId === agent.id}
                isRemoving={removingId === agent.id}
              />
            ))}
          </div>
        )}
      </div>
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
 * Individual agent card component.
 */
function AgentCard({
  agent,
  onSelect,
  onRefresh,
  onRemove,
  isRefreshing,
  isRemoving,
}) {
  const isOnline = agent.status === "online";
  
  // Parse supported modalities from agent card
  const inputModes = agent.defaultInputModes || agent.default_input_modes || [];
  const outputModes = agent.defaultOutputModes || agent.default_output_modes || [];

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-5 hover:border-gray-600 transition-colors">
      <div className="flex items-start justify-between">
        {/* Agent info */}
        <div className="flex items-start gap-4 flex-1">
          <div
            className={`p-3 rounded-lg ${
              isOnline ? "bg-green-600/20" : "bg-red-600/20"
            }`}
          >
            <Bot
              className={`w-8 h-8 ${
                isOnline ? "text-green-400" : "text-red-400"
              }`}
            />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-lg font-semibold truncate">{agent.name}</h3>
              <span
                className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                  isOnline
                    ? "bg-green-600/20 text-green-400"
                    : "bg-red-600/20 text-red-400"
                }`}
              >
                {isOnline ? (
                  <CheckCircle className="w-3 h-3" />
                ) : (
                  <XCircle className="w-3 h-3" />
                )}
                {isOnline ? "Online" : "Offline"}
              </span>
            </div>

            <p className="text-sm text-gray-400 mb-2 line-clamp-2">
              {agent.description || "No description available"}
            </p>

            <div className="flex items-center gap-4 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <ExternalLink className="w-3 h-3" />
                {agent.url}
              </span>
              {agent.version && <span>v{agent.version}</span>}
            </div>

            {/* Supported Modalities */}
            {inputModes.length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-gray-500 mb-1">Supported Input:</p>
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
              <div className="flex flex-wrap gap-2 mt-3">
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
            )}

            {/* Skills */}
            {agent.skills && agent.skills.length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-gray-500 mb-1">Skills:</p>
                <div className="flex flex-wrap gap-2">
                  {agent.skills.slice(0, 4).map((skill) => (
                    <span
                      key={skill.id}
                      className="text-xs px-2 py-1 bg-gray-700 text-gray-300 rounded"
                      title={skill.description}
                    >
                      {skill.name}
                    </span>
                  ))}
                  {agent.skills.length > 4 && (
                    <span className="text-xs px-2 py-1 bg-gray-700 text-gray-400 rounded">
                      +{agent.skills.length - 4} more
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 ml-4">
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh agent status"
          >
            <RefreshCw
              className={`w-4 h-4 ${isRefreshing ? "animate-spin" : ""}`}
            />
          </button>
          <button
            onClick={onRemove}
            disabled={isRemoving}
            className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-600/10 rounded-lg transition-colors disabled:opacity-50"
            title="Remove agent"
          >
            {isRemoving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={onSelect}
            disabled={!isOnline}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              isOnline
                ? "bg-blue-600 hover:bg-blue-500 text-white"
                : "bg-gray-700 text-gray-500 cursor-not-allowed"
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            Chat
          </button>
        </div>
      </div>
    </div>
  );
}
