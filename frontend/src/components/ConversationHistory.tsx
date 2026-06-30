import type { ScenarioConversation } from '../state/scenarioDisplay'

interface ConversationHistoryProps {
  conversations: ScenarioConversation[]
  selectedScenarioId: string
  mobileOpen: boolean
  onSelect: (scenarioId: string) => void
  onCloseMobile: () => void
}

export function ConversationHistory({
  conversations,
  selectedScenarioId,
  mobileOpen,
  onSelect,
  onCloseMobile,
}: ConversationHistoryProps) {
  return (
    <>
      <div
        className={`history-backdrop ${mobileOpen ? 'visible' : ''}`}
        onMouseDown={onCloseMobile}
      />
      <aside
        className={`history-sidebar ${mobileOpen ? 'open' : ''}`}
        aria-label="Conversation history"
      >
        <div className="history-header">
          <p className="product-mark">WHOSE-AGENT</p>
          <button
            type="button"
            className="icon-button history-close"
            aria-label="Close history"
            onClick={onCloseMobile}
          >
            x
          </button>
        </div>
        <p className="history-group-label">Examples</p>
        <nav className="history-list" aria-label="Scenario-backed conversations">
          {conversations.map((conversation) => (
            <button
              type="button"
              className={`history-item ${
                conversation.scenarioId === selectedScenarioId ? 'selected' : ''
              }`}
              aria-current={
                conversation.scenarioId === selectedScenarioId ? 'page' : undefined
              }
              key={conversation.scenarioId}
              onClick={() => onSelect(conversation.scenarioId)}
            >
              <span className="history-title">{conversation.title}</span>
              <span className="history-snippet">{conversation.snippet}</span>
              <span className="history-status">{conversation.statusLabel}</span>
            </button>
          ))}
        </nav>
      </aside>
    </>
  )
}
