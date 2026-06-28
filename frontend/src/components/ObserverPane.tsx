import { finalStatusText, questionForAxis } from '../state/runMachine'
import type {
  CauseProjection,
  CheckerProjection,
  ExplainProjection,
  RunMachineState,
  ScenarioMetadata,
  SubstitutionAxis,
} from '../state/types'

interface ObserverPaneProps {
  state: RunMachineState
  selectedScenario: ScenarioMetadata | null
}

export function ObserverPane({ state, selectedScenario }: ObserverPaneProps) {
  const axis = displayAxis(state.cause, state.checker, selectedScenario)
  const question = questionForAxis(axis)

  return (
    <section className="panel observer-panel" aria-labelledby="observer-title">
      <div className="panel-heading">
        <p className="eyebrow">Boundary observer</p>
        <h2 id="observer-title">{state.cause ? question : 'Observation boundary'}</h2>
      </div>

      {state.status === 'idle' ? (
        <p className="empty-copy">Ready to observe a run.</p>
      ) : null}
      {state.status === 'running' && state.phase === 'plan' ? (
        <p className="empty-copy">Reading delegated context.</p>
      ) : null}
      {state.safeError ? (
        <div className="observer-card error-card">
          <span className="card-kicker">Safe error</span>
          <h3>Observation incomplete</h3>
          <dl>
            <Row label="message" value={state.safeError.message} />
            <Row label="code" value={state.safeError.code ?? 'none'} />
          </dl>
        </div>
      ) : null}
      {state.cause ? <CauseCard cause={state.cause} question={question} /> : null}
      {state.checker ? <CheckerCard checker={state.checker} /> : null}
      {state.explain ? <ExplainCard explain={state.explain} /> : null}
      {state.status === 'completed' ? (
        <div className="final-status" aria-live="polite">
          {finalStatusText(state)}
        </div>
      ) : null}
    </section>
  )
}

function CauseCard({
  cause,
  question,
}: {
  cause: CauseProjection
  question: string
}) {
  return (
    <article className="observer-card reveal-card">
      <span className="card-kicker">Cause</span>
      <h3>{question}</h3>
      <dl>
        <Row label="misreader skill fired" value={String(cause.misreader_skill_fired)} />
        <Row label="selected skill" value={cause.selected_skill_id ?? 'none'} />
        {cause.authority_provenance ? (
          <>
            <Row label="grant status" value={cause.authority_provenance.grant_status} />
            <Row label="result" value={cause.authority_provenance.result} />
            <Row label="action" value={cause.authority_provenance.action_kind} />
            <Row label="target" value={cause.authority_provenance.target} />
          </>
        ) : null}
        {cause.action_attempt_summary ? (
          <Row
            label="action attempted"
            value={String(cause.action_attempt_summary.attempted)}
          />
        ) : null}
      </dl>
    </article>
  )
}

function CheckerCard({ checker }: { checker: CheckerProjection }) {
  return (
    <article className="observer-card checker-card reveal-card">
      <span className="card-kicker">Checker</span>
      <h3>Independent observer</h3>
      <dl>
        <Row label="checker ran" value={String(checker.checker_ran)} />
        <Row label="bypass observed" value={String(checker.checker_observed_bypass)} />
        <Row label="axis" value={checker.substituted ?? 'none'} />
        <Row label="failure mode" value={checker.failure_mode ?? 'none'} />
        <Row label="confidence" value={checker.confidence ?? 'none'} />
        {checker.divergence_summary ? (
          <Row label="divergence" value={checker.divergence_summary} />
        ) : null}
      </dl>
    </article>
  )
}

function ExplainCard({ explain }: { explain: ExplainProjection }) {
  return (
    <article className="observer-card explain-card reveal-card">
      <span className="card-kicker">Explain</span>
      <h3>Agent self-report</h3>
      <dl>
        <Row label="status" value={explain.status} />
        <Row
          label="action"
          value={explain.action_or_adaptation_summary ?? 'not provided'}
        />
        <Row
          label="basis"
          value={explain.treated_as_sufficient_basis ?? 'not provided'}
        />
        <Row
          label="relied on turns"
          value={
            explain.relied_on_turn_indexes.length > 0
              ? explain.relied_on_turn_indexes.join(', ')
              : 'none'
          }
        />
        {explain.rationale_summary ? (
          <Row label="rationale" value={explain.rationale_summary} />
        ) : null}
        {explain.checker_acknowledgement ? (
          <Row label="checker acknowledgement" value={explain.checker_acknowledgement} />
        ) : null}
      </dl>
    </article>
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

function displayAxis(
  cause: CauseProjection | null,
  checker: CheckerProjection | null,
  scenario: ScenarioMetadata | null,
): SubstitutionAxis | null {
  if (scenario?.substitution_axis && scenario.substitution_axis !== 'none') {
    return scenario.substitution_axis
  }
  if (cause?.authority_provenance) {
    return 'authority'
  }
  return checker?.substituted ?? null
}
