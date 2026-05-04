import { Bot, Users, PlusCircle, MessageSquare, ArrowLeft } from "lucide-react";

/**
 * Header component with navigation tabs.
 */
export default function Header({
  activeTab,
  onTabChange,
  selectedAgent,
  onBackToAgents,
}) {
  const tabs = [
    { id: "agents", label: "Agents", icon: Users },
    { id: "register", label: "Register Agent", icon: PlusCircle },
  ];

  return (
    <header className="bg-gray-800 border-b border-gray-700">
      <div className="flex items-center justify-between px-4 py-3">
        {/* Logo / Title */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-600 rounded-lg">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">A2A Chat</h1>
            <p className="text-xs text-gray-400">Multi-Agent Chat Interface</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1">
          {activeTab === "chat" && selectedAgent ? (
            <>
              <button
                onClick={onBackToAgents}
                className="flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
              <div className="flex items-center gap-2 px-3 py-2 bg-blue-600/20 text-blue-400 rounded-lg">
                <MessageSquare className="w-4 h-4" />
                <span className="text-sm font-medium">{selectedAgent.name}</span>
              </div>
            </>
          ) : (
            tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    isActive
                      ? "bg-blue-600 text-white"
                      : "text-gray-300 hover:text-white hover:bg-gray-700"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })
          )}
        </nav>
      </div>
    </header>
  );
}
