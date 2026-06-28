import { EventType } from '@ag-ui/client'
import { describe, expect, it } from 'vitest'
import { runWorkspaceStream } from '../src/state/runCoordinator'
import { initialRunState, runMachine } from '../src/state/runMachine'
import type { RunMachineState, WorkspaceEvent } from '../src/state/types'

describe('runWorkspaceStream', () => {
  it('preserves server RUN_ERROR and skips run reconciliation', async () => {
    let state: RunMachineState = initialRunState('ui_123')
    const dispatch = (event: WorkspaceEvent) => {
      state = runMachine(state, event)
    }
    let fetchRunCalls = 0

    await runWorkspaceStream({
      state,
      selectedScenarioId: '',
      mock: true,
      dispatch,
      runStream: async ({ onEvent }) => {
        onEvent({
          type: EventType.RUN_STARTED,
          threadId: 'ui_123',
          runId: 'run_server',
        })
        onEvent({
          type: EventType.RUN_ERROR,
          message: 'Invalid request.',
          code: 'invalid_request',
        })
      },
      getRun: async () => {
        fetchRunCalls += 1
        throw new Error('fetchRun should not be called')
      },
    })

    expect(fetchRunCalls).toBe(0)
    expect(state.status).toBe('failed')
    expect(state.safeError?.message).toBe('Invalid request.')
    expect(state.safeError?.code).toBe('invalid_request')
    expect(state.safeError?.code).not.toBe('client_stream_failed')
    expect(state.generatedCandidateText).toBe('')
  })
})
