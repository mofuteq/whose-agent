import type { ScenarioConversation } from '../state/scenarioDisplay'
import { boundaryNarrative } from '../state/narrative'
import type { RunMachineState, ScenarioMetadata } from '../state/types'

interface StatusHeaderProps {
  state: RunMachineState
  conversation: ScenarioConversation | null
  selectedScenario: ScenarioMetadata | null
  onOpenHistory: () => void
}

export function StatusHeader({
  state,
  conversation,
  selectedScenario,
  onOpenHistory,
}: StatusHeaderProps) {
  const narrative = boundaryNarrative(state, selectedScenario)

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
        <span>{narrative.headerStatus}</span>
      </div>
    </header>
  )
}
