export type WorkspaceMode = 'conversation' | 'fixed'
export type ServerRunMode = 'prompt_loop' | 'fixed'
export type RunStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool'
export type Phase = 'plan' | 'do' | 'check' | 'explain'
export type TimelinePhase = Phase | 'completed'
export type SubstitutionAxis = 'authority' | 'instruction' | 'role' | 'model' | 'none'

export interface BrowserMessage {
  clientId: string
  role: MessageRole
  content: string
}

export interface ScenarioMetadata {
  scenario_id: string
  selected_skill_id: string | null
  substitution_axis: SubstitutionAxis
  description: string
}

export interface AuthorityProvenanceProjection {
  action_kind: string
  target: string
  prior_agent_proposal_turn: number | null
  principal_grant_turn: number | null
  grant_status: string
  action_attempt_turn: number | null
  result: string
}

export interface ActionAttemptSummary {
  action_kind: string
  target: string
  attempted: boolean
}

export interface CauseProjection {
  misreader_skill_fired: boolean
  selected_skill_id: string | null
  authority_provenance: AuthorityProvenanceProjection | null
  action_attempt_summary: ActionAttemptSummary | null
}

export interface CheckerProjection {
  checker_ran: boolean
  checker_observed_bypass: boolean
  substituted: SubstitutionAxis | null
  failure_mode: string | null
  confidence: string | null
  divergence_summary: string | null
}

export interface ExplainProjection {
  status: string
  action_or_adaptation_summary: string | null
  treated_as_sufficient_basis: string | null
  relied_on_turn_indexes: number[]
  rationale_summary: string | null
  checker_acknowledgement: string | null
}

export interface CompletedProjection {
  run_id: string
  status: 'completed'
  mode: ServerRunMode
  selected_skill_id: string | null
  observation_outcome: string | null
  artifact_names: string[]
}

export interface RunProjection {
  run_id: string
  thread_id: string
  status: Exclude<RunStatus, 'idle'>
  mode: ServerRunMode
  result: CompletedProjection | null
  artifact_names: string[]
  safe_error_code: string | null
}

export interface SafeError {
  message: string
  code: string | null
}

export interface RunMachineState {
  status: RunStatus
  mode: WorkspaceMode
  threadId: string
  serverRunId: string | null
  phase: TimelinePhase | null
  seenPhases: TimelinePhase[]
  messages: BrowserMessage[]
  generatedCandidateText: string
  activeTextMessageId: string | null
  cause: CauseProjection | null
  checker: CheckerProjection | null
  explain: ExplainProjection | null
  completed: CompletedProjection | null
  safeError: SafeError | null
}

export type WorkspaceEvent =
  | { type: 'reset'; threadId: string }
  | { type: 'setMode'; mode: WorkspaceMode }
  | { type: 'setMessages'; messages: BrowserMessage[] }
  | { type: 'startRun'; threadId: string }
  | { type: 'cancelRun' }
  | { type: 'streamEvent'; event: AguiEvent }
  | { type: 'reconcileRun'; run: RunProjection }
  | { type: 'clientFailure'; error: SafeError }

export type AguiEvent =
  | {
      type: 'RUN_STARTED'
      threadId?: string
      runId?: string
    }
  | {
      type: 'TEXT_MESSAGE_START'
      messageId?: string
    }
  | {
      type: 'TEXT_MESSAGE_CONTENT'
      messageId?: string
      delta?: string
    }
  | {
      type: 'TEXT_MESSAGE_END'
      messageId?: string
    }
  | {
      type: 'RUN_ERROR'
      message?: string
      code?: string
    }
  | {
      type: 'RUN_FINISHED'
      threadId?: string
      runId?: string
      result?: unknown
    }
  | {
      type: 'CUSTOM'
      name?: string
      value?: unknown
    }
  | {
      type: string
      [key: string]: unknown
    }
