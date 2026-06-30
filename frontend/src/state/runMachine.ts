import type {
  AguiEvent,
  CauseProjection,
  CheckerProjection,
  CompletedProjection,
  ExplainProjection,
  RunMachineState,
  RunProjection,
  SafeError,
  SubstitutionAxis,
  TimelinePhase,
  WorkspaceEvent,
  WorkspaceMode,
} from './types'

const TIMELINE: TimelinePhase[] = ['plan', 'do', 'check', 'explain', 'completed']

export function newThreadId(): string {
  return `ui_${crypto.randomUUID()}`
}

export function newMessageId(): string {
  return `ui_msg_${crypto.randomUUID()}`
}

export function initialRunState(threadId = newThreadId()): RunMachineState {
  return {
    status: 'idle',
    mode: 'conversation',
    threadId,
    serverRunId: null,
    phase: null,
    seenPhases: [],
    messages: [],
    generatedCandidateText: '',
    activeTextMessageId: null,
    cause: null,
    checker: null,
    explain: null,
    completed: null,
    safeError: null,
  }
}

export function runMachine(
  state: RunMachineState,
  event: WorkspaceEvent,
): RunMachineState {
  switch (event.type) {
    case 'reset':
      return initialRunState(event.threadId)
    case 'setMode':
      return {
        ...state,
        mode: event.mode,
      }
    case 'setMessages':
      return {
        ...state,
        messages: event.messages,
      }
    case 'startRun':
      return {
        ...state,
        status: 'running',
        threadId: event.threadId,
        serverRunId: null,
        phase: null,
        seenPhases: [],
        generatedCandidateText: '',
        activeTextMessageId: null,
        cause: null,
        checker: null,
        explain: null,
        completed: null,
        safeError: null,
      }
    case 'cancelRun':
      return {
        ...state,
        status: 'cancelled',
        activeTextMessageId: null,
        safeError: { message: 'Observation incomplete.', code: 'stream_cancelled' },
      }
    case 'clientFailure':
      return {
        ...state,
        status: 'failed',
        activeTextMessageId: null,
        safeError: event.error,
      }
    case 'streamEvent':
      return applyAguiEvent(state, event.event)
    case 'reconcileRun':
      return reconcileRunProjection(state, event.run)
    default:
      return state
  }
}

export function applyAguiEvent(
  state: RunMachineState,
  event: AguiEvent,
): RunMachineState {
  switch (event.type) {
    case 'RUN_STARTED':
      return {
        ...state,
        status: 'running',
        threadId: stringOrNull(event.threadId) ?? state.threadId,
        serverRunId: stringOrNull(event.runId) ?? state.serverRunId,
      }
    case 'TEXT_MESSAGE_START':
      return {
        ...state,
        activeTextMessageId: stringOrNull(event.messageId),
        generatedCandidateText: '',
      }
    case 'TEXT_MESSAGE_CONTENT':
      if (
        event.messageId === undefined ||
        event.messageId !== state.activeTextMessageId
      ) {
        return state
      }
      return {
        ...state,
        generatedCandidateText: `${state.generatedCandidateText}${event.delta ?? ''}`,
      }
    case 'TEXT_MESSAGE_END':
      if (event.messageId !== state.activeTextMessageId) {
        return state
      }
      return {
        ...state,
        activeTextMessageId: null,
      }
    case 'RUN_ERROR':
      return {
        ...state,
        status: 'failed',
        activeTextMessageId: null,
        safeError: safeErrorFromEvent(event.message, event.code),
      }
    case 'RUN_FINISHED':
      return {
        ...state,
        serverRunId: stringOrNull(event.runId) ?? state.serverRunId,
      }
    case 'CUSTOM':
      return applyCustomEvent(state, stringOrNull(event.name), event.value)
    default:
      return state
  }
}

export function reconcileRunProjection(
  state: RunMachineState,
  run: RunProjection,
): RunMachineState {
  if (state.serverRunId !== run.run_id) {
    return state
  }
  return {
    ...state,
    status: run.status,
    completed: run.result ?? state.completed,
    safeError:
      run.status === 'failed' || run.status === 'cancelled'
        ? {
            message: 'Observation incomplete.',
            code: run.safe_error_code,
          }
        : state.safeError,
  }
}

export function reconciliationRunId(state: RunMachineState): string | null {
  return state.serverRunId
}

export function finalStatusText(state: RunMachineState): string {
  if (state.status === 'failed' || state.status === 'cancelled') {
    return 'Observation incomplete'
  }
  if (state.status !== 'completed') {
    return 'Loading observation'
  }
  if (
    state.cause?.action_attempt_summary?.attempted === true &&
    state.checker?.checker_observed_bypass === true
  ) {
    return 'Boundary drift made visible'
  }
  if (state.checker?.checker_observed_bypass === true) {
    switch (state.checker.substituted) {
      case 'instruction':
        return 'Constraint override observed'
      case 'role':
        return 'Role substitution observed'
      case 'model':
        return 'Model substitution observed'
      case 'authority':
        return 'Boundary drift made visible'
      default:
        return 'Boundary drift made visible'
    }
  }
  return 'No boundary finding'
}

export function currentRunHint(state: RunMachineState): string {
  if (state.status === 'idle') {
    return 'Loading observation'
  }
  if (state.status === 'failed' || state.status === 'cancelled') {
    return 'Observation incomplete'
  }
  if (state.phase === 'plan') {
    return 'Reading delegated context.'
  }
  if (state.phase === 'do') {
    return 'Agent response is forming.'
  }
  if (state.phase === 'check') {
    return 'Checker is observing independently.'
  }
  if (state.phase === 'explain') {
    return 'Agent self-report is available.'
  }
  if (state.phase === 'completed') {
    return finalStatusText(state)
  }
  return 'Observing run state.'
}

export function visibleTimeline(state: RunMachineState): TimelinePhase[] {
  return TIMELINE.filter((phase) => state.seenPhases.includes(phase))
}

function applyCustomEvent(
  state: RunMachineState,
  name: string | null,
  value: unknown,
): RunMachineState {
  if (!isRecord(value)) {
    return state
  }
  switch (name) {
    case 'whose_agent.run.started':
      return {
        ...state,
        serverRunId: stringOrNull(value.run_id) ?? state.serverRunId,
        threadId: stringOrNull(value.thread_id) ?? state.threadId,
        status: 'running',
      }
    case 'whose_agent.phase': {
      const phase = phaseOrNull(value.phase)
      if (phase === null) {
        return state
      }
      return {
        ...state,
        phase,
        seenPhases: addPhase(state.seenPhases, phase),
      }
    }
    case 'whose_agent.cause':
      return {
        ...state,
        cause: normalizeCause(value),
      }
    case 'whose_agent.checker':
      return {
        ...state,
        checker: normalizeChecker(value),
      }
    case 'whose_agent.explain':
      return {
        ...state,
        explain: normalizeExplain(value),
      }
    case 'whose_agent.run.completed': {
      const completed = normalizeCompleted(value)
      return {
        ...state,
        status: 'completed',
        phase: 'completed',
        seenPhases: addPhase(state.seenPhases, 'completed'),
        serverRunId: completed?.run_id ?? state.serverRunId,
        completed,
      }
    }
    default:
      return state
  }
}

function addPhase(
  seenPhases: TimelinePhase[],
  phase: TimelinePhase,
): TimelinePhase[] {
  if (seenPhases.includes(phase)) {
    return seenPhases
  }
  return [...seenPhases, phase].sort(
    (left, right) => TIMELINE.indexOf(left) - TIMELINE.indexOf(right),
  )
}

function safeErrorFromEvent(message: unknown, code: unknown): SafeError {
  return {
    message: stringOrNull(message) ?? 'Observation incomplete.',
    code: stringOrNull(code),
  }
}

function normalizeCause(value: Record<string, unknown>): CauseProjection {
  return {
    misreader_skill_fired: value.misreader_skill_fired === true,
    selected_skill_id: stringOrNull(value.selected_skill_id),
    authority_provenance: isRecord(value.authority_provenance)
      ? {
          action_kind: stringOrEmpty(value.authority_provenance.action_kind),
          target: stringOrEmpty(value.authority_provenance.target),
          prior_agent_proposal_turn: numberOrNull(
            value.authority_provenance.prior_agent_proposal_turn,
          ),
          principal_grant_turn: numberOrNull(
            value.authority_provenance.principal_grant_turn,
          ),
          grant_status: stringOrEmpty(value.authority_provenance.grant_status),
          action_attempt_turn: numberOrNull(
            value.authority_provenance.action_attempt_turn,
          ),
          result: stringOrEmpty(value.authority_provenance.result),
        }
      : null,
    action_attempt_summary: isRecord(value.action_attempt_summary)
      ? {
          action_kind: stringOrEmpty(value.action_attempt_summary.action_kind),
          target: stringOrEmpty(value.action_attempt_summary.target),
          attempted: value.action_attempt_summary.attempted === true,
        }
      : null,
  }
}

function normalizeChecker(value: Record<string, unknown>): CheckerProjection {
  return {
    checker_ran: value.checker_ran === true,
    checker_observed_bypass: value.checker_observed_bypass === true,
    substituted: axisOrNull(value.substituted),
    failure_mode: stringOrNull(value.failure_mode),
    confidence: stringOrNull(value.confidence),
    divergence_summary: stringOrNull(value.divergence_summary),
  }
}

function normalizeExplain(value: Record<string, unknown>): ExplainProjection {
  return {
    status: stringOrEmpty(value.status),
    action_or_adaptation_summary: stringOrNull(
      value.action_or_adaptation_summary,
    ),
    treated_as_sufficient_basis: stringOrNull(value.treated_as_sufficient_basis),
    relied_on_turn_indexes: Array.isArray(value.relied_on_turn_indexes)
      ? value.relied_on_turn_indexes.filter(
          (item): item is number => typeof item === 'number',
        )
      : [],
    rationale_summary: stringOrNull(value.rationale_summary),
    checker_acknowledgement: stringOrNull(value.checker_acknowledgement),
  }
}

function normalizeCompleted(
  value: Record<string, unknown>,
): CompletedProjection | null {
  const runId = stringOrNull(value.run_id)
  if (runId === null) {
    return null
  }
  return {
    run_id: runId,
    status: 'completed',
    mode: value.mode === 'fixed' ? 'fixed' : 'prompt_loop',
    selected_skill_id: stringOrNull(value.selected_skill_id),
    observation_outcome: stringOrNull(value.observation_outcome),
    artifact_names: Array.isArray(value.artifact_names)
      ? value.artifact_names.filter(
          (item): item is string => typeof item === 'string',
        )
      : [],
  }
}

function phaseOrNull(value: unknown): TimelinePhase | null {
  return value === 'plan' ||
    value === 'do' ||
    value === 'check' ||
    value === 'explain'
    ? value
    : null
}

function axisOrNull(value: unknown): SubstitutionAxis | null {
  return value === 'authority' ||
    value === 'instruction' ||
    value === 'role' ||
    value === 'model' ||
    value === 'none'
    ? value
    : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function stringOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

export function serverModeFromWorkspace(mode: WorkspaceMode): 'fixed' | 'prompt_loop' {
  return mode === 'fixed' ? 'fixed' : 'prompt_loop'
}
