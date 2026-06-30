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
    expect(narrative.statusDetail).toBe('Checker-only observation')
    expect(narrative.interruptionBody).toContain(
      'did not report an action attempt',
    )
    expect(narrative.interruptionBody).not.toContain(
      'This action was not authorized by you.',
    )
  })
})
