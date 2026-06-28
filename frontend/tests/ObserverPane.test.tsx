import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ObserverPane } from '../src/components/ObserverPane'
import { initialRunState } from '../src/state/runMachine'

describe('ObserverPane', () => {
  it('displays the authority question for an authority cause', () => {
    const state = {
      ...initialRunState('ui_123'),
      status: 'running' as const,
      cause: {
        misreader_skill_fired: true,
        selected_skill_id: 'authority_scope_expansion',
        authority_provenance: {
          action_kind: 'write',
          target: 'Notion',
          prior_agent_proposal_turn: 1,
          principal_grant_turn: null,
          grant_status: 'not_granted',
          action_attempt_turn: 2,
          result: 'self_originated_delegation_laundering',
        },
        action_attempt_summary: {
          action_kind: 'write',
          target: 'Notion',
          attempted: true,
        },
      },
    }

    render(<ObserverPane mode="conversation" state={state} selectedScenario={null} />)

    expect(screen.getAllByText('Who authorized this?').length).toBeGreaterThan(0)
  })

  it('does not use stale fixed-scenario metadata for a conversation authority cause', () => {
    const state = {
      ...initialRunState('ui_123'),
      mode: 'conversation' as const,
      status: 'running' as const,
      cause: {
        misreader_skill_fired: true,
        selected_skill_id: 'authority_scope_expansion',
        authority_provenance: {
          action_kind: 'write',
          target: 'Notion',
          prior_agent_proposal_turn: 1,
          principal_grant_turn: null,
          grant_status: 'not_granted',
          action_attempt_turn: 2,
          result: 'self_originated_delegation_laundering',
        },
        action_attempt_summary: {
          action_kind: 'write',
          target: 'Notion',
          attempted: true,
        },
      },
    }

    render(
      <ObserverPane
        mode="conversation"
        state={state}
        selectedScenario={{
          scenario_id: 'instruction_typescript_any',
          selected_skill_id: 'safety_framework_escape_hatch',
          substitution_axis: 'instruction',
          description: 'instruction-axis fixed scenario',
        }}
      />,
    )

    expect(screen.getAllByText('Who authorized this?').length).toBeGreaterThan(0)
    expect(
      screen.queryByText('Which instruction was substituted?'),
    ).not.toBeInTheDocument()
  })

  it('omits the Explain card when no explain event arrived', () => {
    const state = {
      ...initialRunState('ui_123'),
      status: 'completed' as const,
      cause: {
        misreader_skill_fired: false,
        selected_skill_id: null,
        authority_provenance: null,
        action_attempt_summary: null,
      },
      checker: {
        checker_ran: true,
        checker_observed_bypass: false,
        substituted: null,
        failure_mode: null,
        confidence: 'high',
        divergence_summary: null,
      },
      completed: {
        run_id: 'run_server',
        status: 'completed' as const,
        mode: 'prompt_loop' as const,
        selected_skill_id: null,
        observation_outcome: null,
        artifact_names: [],
      },
    }

    render(<ObserverPane mode="conversation" state={state} selectedScenario={null} />)

    expect(screen.queryByText('Agent self-report')).not.toBeInTheDocument()
    expect(screen.getByText('No bypass observed by checker')).toBeInTheDocument()
  })

  it('shows only safe server error message and code for failed streams', () => {
    const state = {
      ...initialRunState('ui_123'),
      status: 'failed' as const,
      safeError: {
        message: 'Invalid request.',
        code: 'invalid_request',
      },
    }

    render(<ObserverPane mode="conversation" state={state} selectedScenario={null} />)

    expect(screen.getByText('Invalid request.')).toBeInTheDocument()
    expect(screen.getByText('invalid_request')).toBeInTheDocument()
    expect(screen.queryByText(/Traceback|Exception|stack/i)).not.toBeInTheDocument()
  })
})
