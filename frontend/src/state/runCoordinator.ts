import { runAguiStream } from '../api/aguiStream'
import { fetchRun } from '../api/runs'
import {
  newThreadId,
  serverModeFromWorkspace,
} from './runMachine'
import type { AguiEvent, RunMachineState, WorkspaceEvent } from './types'

type RunStream = typeof runAguiStream
type FetchRun = typeof fetchRun

export interface RunCoordinatorRequest {
  state: RunMachineState
  selectedScenarioId: string
  mock: boolean
  signal?: AbortSignal
  dispatch: (event: WorkspaceEvent) => void
  runStream?: RunStream
  getRun?: FetchRun
}

export async function runWorkspaceStream({
  state,
  selectedScenarioId,
  mock,
  signal,
  dispatch,
  runStream = runAguiStream,
  getRun = fetchRun,
}: RunCoordinatorRequest): Promise<void> {
  const threadId = state.threadId || newThreadId()
  let latestServerRunId: string | null = null
  let receivedServerRunError = false

  dispatch({ type: 'startRun', threadId })

  try {
    await runStream({
      threadId,
      mode: serverModeFromWorkspace(state.mode),
      scenarioId: selectedScenarioId,
      mock,
      messages: state.messages,
      signal,
      onEvent: (event) => {
        const aguiEvent = event as AguiEvent
        latestServerRunId = serverRunIdFromEvent(aguiEvent) ?? latestServerRunId
        if (aguiEvent.type === 'RUN_ERROR') {
          receivedServerRunError = true
        }
        dispatch({ type: 'streamEvent', event: aguiEvent })
      },
    })

    if (!receivedServerRunError && latestServerRunId !== null) {
      const publicRun = await getRun(latestServerRunId)
      dispatch({ type: 'reconcileRun', run: publicRun })
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      dispatch({ type: 'cancelRun' })
      return
    }
    if (!receivedServerRunError) {
      dispatch({
        type: 'clientFailure',
        error: {
          message: 'Observation incomplete.',
          code: 'client_stream_failed',
        },
      })
    }
  }
}

function serverRunIdFromEvent(event: AguiEvent): string | null {
  if (
    (event.type === 'RUN_STARTED' || event.type === 'RUN_FINISHED') &&
    typeof event.runId === 'string'
  ) {
    return event.runId
  }
  if (
    event.type === 'CUSTOM' &&
    (event.name === 'whose_agent.run.started' ||
      event.name === 'whose_agent.run.completed') &&
    isRecord(event.value) &&
    typeof event.value.run_id === 'string'
  ) {
    return event.value.run_id
  }
  return null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
