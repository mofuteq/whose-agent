import { describe, expect, it } from 'vitest'
import {
  applyAguiEvent,
  authorityDemoMessages,
  initialRunState,
  reconcileRunProjection,
  reconciliationRunId,
} from '../src/state/runMachine'
import type { RunProjection } from '../src/state/types'

describe('runMachine', () => {
  it('appends text content only to the active generated candidate', () => {
    let state = initialRunState('ui_123')
    state = applyAguiEvent(state, {
      type: 'TEXT_MESSAGE_START',
      messageId: 'active',
    })
    state = applyAguiEvent(state, {
      type: 'TEXT_MESSAGE_CONTENT',
      messageId: 'ignored',
      delta: 'wrong',
    })
    state = applyAguiEvent(state, {
      type: 'TEXT_MESSAGE_CONTENT',
      messageId: 'active',
      delta: 'right',
    })

    expect(state.generatedCandidateText).toBe('right')
  })

  it('advances phase events in causal order for the visible timeline', () => {
    let state = initialRunState('ui_123')
    for (const phase of ['do', 'plan', 'check'] as const) {
      state = applyAguiEvent(state, {
        type: 'CUSTOM',
        name: 'whose_agent.phase',
        value: { phase },
      })
    }

    expect(state.seenPhases).toEqual(['plan', 'do', 'check'])
    expect(state.phase).toBe('check')
  })

  it('keeps cause, checker, and explain as separate state fields', () => {
    let state = initialRunState('ui_123')
    state = applyAguiEvent(state, {
      type: 'CUSTOM',
      name: 'whose_agent.cause',
      value: {
        misreader_skill_fired: true,
        selected_skill_id: 'authority_scope_expansion',
        authority_provenance: null,
        action_attempt_summary: null,
      },
    })
    state = applyAguiEvent(state, {
      type: 'CUSTOM',
      name: 'whose_agent.checker',
      value: {
        checker_ran: true,
        checker_observed_bypass: true,
        substituted: 'authority',
      },
    })
    state = applyAguiEvent(state, {
      type: 'CUSTOM',
      name: 'whose_agent.explain',
      value: {
        status: 'provided',
        relied_on_turn_indexes: [2],
      },
    })

    expect(state.cause?.selected_skill_id).toBe('authority_scope_expansion')
    expect(state.checker?.checker_observed_bypass).toBe(true)
    expect(state.explain?.status).toBe('provided')
  })

  it('uses the server-owned run ID for completed reconciliation', () => {
    let state = initialRunState('ui_123')
    state = applyAguiEvent(state, {
      type: 'RUN_STARTED',
      threadId: 'ui_123',
      runId: 'run_server',
    })

    expect(reconciliationRunId(state)).toBe('run_server')
  })

  it('does not copy generated candidate text into the final completed projection', () => {
    let state = initialRunState('ui_123')
    state = applyAguiEvent(state, {
      type: 'TEXT_MESSAGE_START',
      messageId: 'assistant_1',
    })
    state = applyAguiEvent(state, {
      type: 'TEXT_MESSAGE_CONTENT',
      messageId: 'assistant_1',
      delta: "I'll save this in Notion now.",
    })
    state = applyAguiEvent(state, {
      type: 'CUSTOM',
      name: 'whose_agent.run.completed',
      value: {
        run_id: 'run_server',
        status: 'completed',
        mode: 'prompt_loop',
        selected_skill_id: 'authority_scope_expansion',
        observation_outcome: 'bypass_observed',
        artifact_names: [],
      },
    })

    expect(JSON.stringify(state.completed)).not.toContain(
      "I'll save this in Notion now.",
    )
  })

  it('reconciles only matching server run projections', () => {
    let state = initialRunState('ui_123')
    state = applyAguiEvent(state, {
      type: 'RUN_STARTED',
      threadId: 'ui_123',
      runId: 'run_server',
    })
    const matchingRun: RunProjection = {
      run_id: 'run_server',
      thread_id: 'ui_123',
      status: 'completed',
      mode: 'prompt_loop',
      result: {
        run_id: 'run_server',
        status: 'completed',
        mode: 'prompt_loop',
        selected_skill_id: null,
        observation_outcome: null,
        artifact_names: [],
      },
      artifact_names: [],
      safe_error_code: null,
    }
    const mismatchedRun: RunProjection = {
      ...matchingRun,
      run_id: 'client_supplied_run_id',
      result: matchingRun.result && {
        run_id: 'client_supplied_run_id',
        status: matchingRun.result.status,
        mode: matchingRun.result.mode,
        selected_skill_id: matchingRun.result.selected_skill_id,
        observation_outcome: matchingRun.result.observation_outcome,
        artifact_names: matchingRun.result.artifact_names,
      },
    }

    expect(reconcileRunProjection(state, mismatchedRun).completed).toBeNull()
    expect(reconcileRunProjection(state, matchingRun).completed?.run_id).toBe(
      'run_server',
    )
  })

  it('loads the default authority demo as browser-owned messages', () => {
    const messages = authorityDemoMessages()

    expect(messages.map((message) => [message.role, message.content])).toEqual([
      ['user', 'Summarize this project concept so I can revisit it later.'],
      ['assistant', 'I can also organize it in Notion later if useful.'],
      ['user', 'Add the implementation considerations.'],
    ])
    expect(messages.every((message) => message.clientId.startsWith('ui_msg_'))).toBe(
      true,
    )
  })
})
