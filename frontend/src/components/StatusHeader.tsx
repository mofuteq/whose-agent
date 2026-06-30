import type { ScenarioConversation } from '../state/conversationExamples'
import { boundaryNarrative } from '../state/narrative'
import type { RunMachineState } from '../state/types'

interface StatusHeaderProps {
  state: RunMachineState
  conversation: ScenarioConversation | null
  onOpenHistory: () => void
}

export function StatusHeader({
  state,
  conversation,
  onOpenHistory,
}: StatusHeaderProps) {
  const narrative = boundaryNarrative(state)
  const statusText =
    state.status === 'running'
      ? 'Observing'
      : conversation?.statusLabel ?? narrative.statusDetail

  return (
    <header className="top-bar">
      <button
        type="button"
        className="secondary-action history-toggle"
        onClick={onOpenHistory}
      >
        History
      </button>
      <div className="conversation-title-block">
        <p className="eyebrow">Conversation</p>
        <h1>{conversation?.title ?? 'Loading conversation'}</h1>
      </div>
      <div className="run-state" aria-live="polite">
        <span className={`status-dot status-${state.status}`} aria-hidden="true" />
        <span>{statusText}</span>
      </div>
    </header>
  )
}
