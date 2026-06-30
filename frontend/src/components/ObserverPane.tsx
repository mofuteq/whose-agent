import { useEffect, useRef } from 'react'
import {
  actionAttempted,
  actionTarget,
  boundaryNarrative,
} from '../state/narrative'
import type { RunMachineState, ScenarioMetadata, WorkspaceMode } from '../state/types'

interface ObserverPaneProps {
  isOpen: boolean
  mode: WorkspaceMode
  state: RunMachineState
  selectedScenario: ScenarioMetadata | null
  onClose: () => void
}

export function ObserverPane({
  isOpen,
  mode,
  state,
  selectedScenario,
  onClose,
}: ObserverPaneProps) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null)
  const narrative = boundaryNarrative(state)

  useEffect(() => {
    if (!isOpen) {
      return undefined
    }
    closeButtonRef.current?.focus()
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [isOpen, onClose])

  if (!isOpen) {
    return null
  }

  return (
    <div className="drawer-scrim" onMouseDown={onClose}>
      <aside
        className="inspector-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="inspector-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="drawer-header">
          <div>
            <p className="eyebrow">Observer</p>
            <h2 id="inspector-title">Why was this flagged?</h2>
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Close inspector"
            onClick={onClose}
            ref={closeButtonRef}
          >
            x
          </button>
        </header>

        <section className="inspector-section primary-answer">
          <p className="question">{narrative.question}</p>
          <p className="answer">{narrative.answer}</p>
        </section>

        <section className="inspector-section">
          <h3>What the assistant relied on</h3>
          <p>{narrative.reliance}</p>
        </section>

        <section className="inspector-section">
          <h3>Independent check</h3>
          <p>{narrative.independentCheck}</p>
        </section>

        {narrative.selfReport ? (
          <section className="inspector-section">
            <h3>What the assistant says it believed</h3>
            <p className="quoted">"{narrative.selfReport}"</p>
          </section>
        ) : null}

        <section className="inspector-section sequence-section">
          <SequenceStep
            term="Cause"
            label="What the assistant relied on"
            value={narrative.reliance}
          />
          <span className="sequence-arrow" aria-hidden="true" />
          <SequenceStep
            term="Checker"
            label="What an independent observer found"
            value={narrative.independentCheck}
          />
          <span className="sequence-arrow" aria-hidden="true" />
          <SequenceStep
            term="Explain"
            label="What the assistant says it believed"
            value={narrative.selfReport ?? 'No self-explanation is available.'}
          />
        </section>

        <details className="technical-trace">
          <summary>Technical trace</summary>
          <dl>
            <Row
              label="selected skill"
              value={
                state.cause?.selected_skill_id ??
                state.completed?.selected_skill_id ??
                'none'
              }
            />
            <Row
              label="grant status"
              value={state.cause?.authority_provenance?.grant_status ?? 'not available'}
            />
            <Row
              label="authority provenance"
              value={state.cause?.authority_provenance?.result ?? 'not available'}
            />
            <Row
              label="action attempted"
              value={actionAttemptText(state)}
            />
            <Row
              label="action attempted target"
              value={actionTarget(state)}
            />
            <Row
              label="checker confidence"
              value={state.checker?.confidence ?? 'not available'}
            />
            <Row
              label="failure mode"
              value={state.checker?.failure_mode ?? 'none'}
            />
            <Row
              label="checker evidence"
              value={state.checker?.divergence_summary ?? 'not available'}
            />
            <Row
              label="run ID"
              value={state.completed?.run_id ?? state.serverRunId ?? 'not available'}
            />
            <Row
              label="run mode"
              value={runModeText(state, mode)}
            />
            <Row
              label="current example"
              value={
                mode === 'fixed'
                  ? selectedScenario?.scenario_id ?? 'unknown fixed scenario'
                  : 'authority demo'
              }
            />
            <Row
              label="trace phases"
              value={
                state.seenPhases.length > 0
                  ? state.seenPhases.join(' -> ')
                  : 'none'
              }
            />
          </dl>
        </details>
      </aside>
    </div>
  )
}

function SequenceStep({
  term,
  label,
  value,
}: {
  term: string
  label: string
  value: string
}) {
  return (
    <div className="sequence-step">
      <span>{term}</span>
      <h3>{label}</h3>
      <p>{value}</p>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  )
}

function actionAttemptText(state: RunMachineState): string {
  if (!actionAttempted(state)) {
    return 'no reported action attempt'
  }
  const actionKind = state.cause?.action_attempt_summary?.action_kind ?? 'action'
  return `${actionKind} reported`
}

function runModeText(state: RunMachineState, mode: WorkspaceMode): string {
  if (state.completed?.mode === 'fixed') {
    return 'fixed scenario'
  }
  if (state.completed?.mode === 'prompt_loop') {
    return 'conversation'
  }
  return mode === 'fixed' ? 'fixed scenario' : 'conversation'
}
