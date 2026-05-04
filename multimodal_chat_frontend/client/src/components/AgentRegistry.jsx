import { useState } from "react";
import {
  Bot,
  Link,
  FileText,
  Loader2,
  CheckCircle,
  AlertCircle,
  Search,
  Image,
  Video,
  Zap,
} from "lucide-react";
import * as api from "../api";

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
 * Agent registration tab - allows registering new A2A agents.
 */
export default function AgentRegistry({ onRegister, onSuccess }) {
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [fetchedCard, setFetchedCard] = useState(null);

  // Fetch agent card to preview before registering
  const handleFetchCard = async () => {
    if (!url.trim()) {
      setError("Please enter an agent URL");
      return;
    }

    setIsFetching(true);
    setError(null);
    setFetchedCard(null);

    try {
      const result = await api.fetchAgentCard(url.trim());
      setFetchedCard(result.agentCard);
      
      // Auto-fill name and description from agent card
      if (result.agentCard.name && !name) {
        setName(result.agentCard.name);
      }
      if (result.agentCard.description && !description) {
        setDescription(result.agentCard.description);
      }
    } catch (err) {
      setError(`Failed to fetch agent card: ${err.message}`);
    } finally {
      setIsFetching(false);
    }
  };

  // Register the agent
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!url.trim()) {
      setError("Agent URL is required");
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      await onRegister({
        url: url.trim(),
        name: name.trim() || undefined,
        description: description.trim() || undefined,
      });

      setSuccess(true);
      setUrl("");
      setName("");
      setDescription("");
      setFetchedCard(null);

      // Navigate back to agents list after short delay
      setTimeout(() => {
        onSuccess();
      }, 1500);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Parse modalities from fetched card
  const inputModes = fetchedCard?.defaultInputModes || fetchedCard?.default_input_modes || [];

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold">Register New Agent</h2>
          <p className="text-gray-400 mt-1">
            Add a new A2A-compatible agent to your registry
          </p>
        </div>

        {/* Registration form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* URL input with fetch button */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Agent URL <span className="text-red-400">*</span>
            </label>
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Link className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="http://localhost:10010"
                  className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-blue-500 transition-colors"
                />
              </div>
              <button
                type="button"
                onClick={handleFetchCard}
                disabled={isFetching || !url.trim()}
                className="flex items-center gap-2 px-4 py-3 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 rounded-lg transition-colors"
              >
                {isFetching ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Search className="w-5 h-5" />
                )}
                Fetch
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Enter the base URL of the A2A agent (e.g., http://localhost:10010)
            </p>
          </div>

          {/* Fetched agent card preview */}
          {fetchedCard && (
            <div className="p-4 bg-gray-800 border border-green-600/30 rounded-lg">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-green-600/20 rounded-lg">
                  <Bot className="w-6 h-6 text-green-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold">{fetchedCard.name}</h4>
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  </div>
                  <p className="text-sm text-gray-400 mb-2">
                    {fetchedCard.description}
                  </p>
                  {fetchedCard.version && (
                    <span className="text-xs text-gray-500">
                      Version: {fetchedCard.version}
                    </span>
                  )}
                  
                  {/* Supported Modalities */}
                  {inputModes.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 mb-1">Supported Input:</p>
                      <div className="flex flex-wrap gap-1">
                        {inputModes.map((mode) => {
                          const Icon = getModalityIcon(mode);
                          return (
                            <span
                              key={mode}
                              className="flex items-center gap-1 text-xs px-2 py-0.5 bg-blue-600/20 text-blue-400 rounded"
                            >
                              <Icon className="w-3 h-3" />
                              {mode}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  
                  {/* Skills preview */}
                  {fetchedCard.skills && fetchedCard.skills.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 mb-1">Skills:</p>
                      <div className="flex flex-wrap gap-1">
                        {fetchedCard.skills.map((skill) => (
                          <span
                            key={skill.id}
                            className="text-xs px-2 py-0.5 bg-gray-700 text-gray-300 rounded"
                          >
                            {skill.name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Name input */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Display Name
            </label>
            <div className="relative">
              <Bot className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Agent (auto-filled from agent card)"
                className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Optional. If not provided, the name from the agent card will be used.
            </p>
          </div>

          {/* Description input */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Description
            </label>
            <div className="relative">
              <FileText className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="A brief description of this agent (auto-filled from agent card)"
                rows={3}
                className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-blue-500 transition-colors resize-none"
              />
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div className="flex items-center gap-2 p-4 bg-red-600/10 border border-red-600/30 rounded-lg text-red-400">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Success message */}
          {success && (
            <div className="flex items-center gap-2 p-4 bg-green-600/10 border border-green-600/30 rounded-lg text-green-400">
              <CheckCircle className="w-5 h-5 flex-shrink-0" />
              <p className="text-sm">Agent registered successfully! Redirecting...</p>
            </div>
          )}

          {/* Submit button */}
          <button
            type="submit"
            disabled={isLoading || !url.trim()}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-medium transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Registering...
              </>
            ) : (
              <>
                <Bot className="w-5 h-5" />
                Register Agent
              </>
            )}
          </button>
        </form>

        {/* Help text */}
        <div className="mt-8 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
          <h4 className="font-medium mb-2">What is an A2A Agent?</h4>
          <p className="text-sm text-gray-400">
            A2A (Agent-to-Agent) is a protocol for communication between AI agents.
            Any A2A-compatible agent exposes an agent card at{" "}
            <code className="text-blue-400">/.well-known/agent.json</code> that
            describes its capabilities, skills, and supported input/output modes.
          </p>
        </div>
      </div>
    </div>
  );
}
