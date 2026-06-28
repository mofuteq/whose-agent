import { currentRunHint } from '../state/runMachine'
import type { RunMachineState } from '../state/types'

interface StatusHeaderProps {
  state: RunMachineState
}

export function StatusHeader({ state }: StatusHeaderProps) {
  return (
    <header className="status-header">
      <div>
        <p className="eyebrow">whose-agent</p>
        <h1>Observation Workspace</h1>
      </div>
      <div className="run-state" aria-live="polite">
        <span className={`status-dot status-${state.status}`} aria-hidden="true" />
        <span>{state.status}</span>
        <span className="run-hint">{currentRunHint(state)}</span>
      </div>
    </header>
  )
}
