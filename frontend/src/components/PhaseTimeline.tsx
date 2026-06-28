import type { RunMachineState, TimelinePhase } from '../state/types'

const PHASES: TimelinePhase[] = ['plan', 'do', 'check', 'explain', 'completed']

interface PhaseTimelineProps {
  state: RunMachineState
}

export function PhaseTimeline({ state }: PhaseTimelineProps) {
  return (
    <section className="phase-timeline" aria-label="Run phase timeline">
      {PHASES.map((phase) => {
        const seen = state.seenPhases.includes(phase)
        const active = state.phase === phase
        return (
          <div
            className={`timeline-node ${seen ? 'seen' : ''} ${active ? 'active' : ''}`}
            key={phase}
          >
            <span className="timeline-dot" aria-hidden="true" />
            <span>{phase}</span>
          </div>
        )
      })}
    </section>
  )
}
