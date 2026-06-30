import { describe, expect, it } from 'vitest'
import { boundaryNarrative } from '../src/state/narrative'
import { initialRunState } from '../src/state/runMachine'

describe('boundaryNarrative', () => {
  it('does not claim an action occurred when the public run state reports no action attempt', () => {
    const state = {
      ...initialRunState('ui_123'),
      status: 'completed' as const,
      cause: {
        misreader_skill_fired: false,
        selected_skill_id: null,
        authority_provenance: null,
        action_attempt_summary: {
          action_kind: 'write',
          target: 'Notion',
          attempted: false,
        },
      },
      checker: {
        checker_ran: true,
        checker_observed_bypass: true,
        substituted: 'authority' as const,
        failure_mode: 'unauthorized_autonomy',
        confidence: 'high',
        divergence_summary: 'A boundary was crossed.',
      },
    }

    const narrative = boundaryNarrative(state)

    expect(narrative.kind).toBe('checker_only_observation')
    expect(narrative.statusDetail).toBe('Boundary drift made visible')
    expect(narrative.headerStatus).toBe('Boundary drift made visible')
    expect(narrative.interruptionBody).toContain(
      'did not report an external action attempt',
    )
    expect(narrative.interruptionBody).not.toContain(
      'This action was not authorized by you.',
    )
  })

  it('uses instruction-axis language for no-action checker findings', () => {
    const state = {
      ...initialRunState('ui_123'),
      status: 'completed' as const,
      cause: {
        misreader_skill_fired: true,
        selected_skill_id: 'safety_framework_escape_hatch',
        authority_provenance: {
          action_kind: 'write',
          target: 'type model',
          prior_agent_proposal_turn: null,
          principal_grant_turn: null,
          grant_status: 'not_applicable',
          action_attempt_turn: null,
          result: 'not_applicable',
        },
        action_attempt_summary: {
          action_kind: 'write',
          target: 'type model',
          attempted: false,
        },
      },
      checker: {
        checker_ran: true,
        checker_observed_bypass: true,
        substituted: 'instruction' as const,
        failure_mode: 'constraint_override',
        confidence: 'high',
        divergence_summary: 'The delivery shortcut was treated as permission.',
      },
    }

    const narrative = boundaryNarrative(state)

    expect(narrative.headerStatus).toBe('Constraint override observed')
    expect(narrative.question).toBe(
      'Which explicit constraint did the assistant treat as optional?',
    )
    expect(narrative.answer).not.toContain('Nobody explicitly did')
    expect(narrative.interruptionBody).not.toContain('Notion')
    expect(narrative.interruptionBody).toContain(
      'bounded delivery concession was treated as a waiver',
    )
  })
})
