import type { ScenarioConversation } from '../state/scenarioDisplay'
import type { PromptLoopPresetMetadata } from '../state/types'

interface ConversationHistoryProps {
  presets: PromptLoopPresetMetadata[]
  conversations: ScenarioConversation[]
  selectedKey: string
  mobileOpen: boolean
  onSelectPreset: (presetId: string) => void
  onSelectCustom: () => void
  onSelectFixed: (scenarioId: string) => void
  onCloseMobile: () => void
}

export function ConversationHistory({
  presets,
  conversations,
  selectedKey,
  mobileOpen,
  onSelectPreset,
  onSelectCustom,
  onSelectFixed,
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
        <p className="history-group-label">Conversation starters</p>
        <nav className="history-list" aria-label="Conversation starters">
          {presets.map((preset) => (
            <button
              type="button"
              className={`history-item ${
                selectedKey === presetKey(preset.preset_id) ? 'selected' : ''
              }`}
              aria-current={
                selectedKey === presetKey(preset.preset_id) ? 'page' : undefined
              }
              key={preset.preset_id}
              onClick={() => onSelectPreset(preset.preset_id)}
            >
              <span className="history-title">{preset.display_title}</span>
              <span className="history-snippet">{preset.description}</span>
              <span className="history-status">
                {preset.prior_completed_agent_turns} prior agent turn
                {preset.prior_completed_agent_turns === 1 ? '' : 's'}
              </span>
            </button>
          ))}
          <button
            type="button"
            className={`history-item ${
              selectedKey === 'custom:new' ? 'selected' : ''
            }`}
            aria-current={selectedKey === 'custom:new' ? 'page' : undefined}
            onClick={onSelectCustom}
          >
            <span className="history-title">New custom observation</span>
            <span className="history-snippet">Start from an empty prompt.</span>
            <span className="history-status">Live prompt</span>
          </button>
        </nav>
        <p className="history-group-label fixed-history-label">
          Fixed scenario playback
        </p>
        <nav className="history-list" aria-label="Fixed scenario playback">
          {conversations.map((conversation) => (
            <button
              type="button"
              className={`history-item ${
                selectedKey === fixedKey(conversation.scenarioId) ? 'selected' : ''
              }`}
              aria-current={
                selectedKey === fixedKey(conversation.scenarioId) ? 'page' : undefined
              }
              key={conversation.scenarioId}
              onClick={() => onSelectFixed(conversation.scenarioId)}
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

function presetKey(presetId: string): string {
  return `preset:${presetId}`
}

function fixedKey(scenarioId: string): string {
  return `fixed:${scenarioId}`
}
