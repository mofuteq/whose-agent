import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { runAguiStream } from './api/aguiStream'
import { fetchRun } from './api/runs'
import { fetchScenarios } from './api/scenarios'
import { ConversationPane } from './components/ConversationPane'
import { ObserverPane } from './components/ObserverPane'
import { PhaseTimeline } from './components/PhaseTimeline'
import { RunComposer } from './components/RunComposer'
import { StatusHeader } from './components/StatusHeader'
import {
  authorityDemoMessages,
  initialRunState,
  newMessageId,
  newThreadId,
  reconciliationRunId,
  runMachine,
  serverModeFromWorkspace,
} from './state/runMachine'
import type { AguiEvent, BrowserMessage, ScenarioMetadata } from './state/types'

function App() {
  const [state, dispatch] = useReducer(runMachine, undefined, () => initialRunState())
  const [scenarios, setScenarios] = useState<ScenarioMetadata[]>([])
  const [selectedScenarioId, setSelectedScenarioId] = useState('')
  const [mock, setMock] = useState(true)
  const abortRef = useRef<AbortController | null>(null)
  const latestServerRunId = useRef<string | null>(null)

  useEffect(() => {
    let active = true
    fetchScenarios()
      .then((loadedScenarios) => {
        if (!active) {
          return
        }
        setScenarios(loadedScenarios)
        setSelectedScenarioId((current) => current || loadedScenarios[0]?.scenario_id || '')
      })
      .catch(() => {
        if (active) {
          setScenarios([])
        }
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    latestServerRunId.current = state.serverRunId
  }, [state.serverRunId])

  const selectedScenario = useMemo(
    () =>
      scenarios.find((scenario) => scenario.scenario_id === selectedScenarioId) ??
      null,
    [scenarios, selectedScenarioId],
  )

  function setMessages(messages: BrowserMessage[]) {
    dispatch({ type: 'setMessages', messages })
  }

  function loadAuthorityDemo() {
    dispatch({ type: 'setMode', mode: 'conversation' })
    setMessages(authorityDemoMessages())
  }

  function addTurn() {
    setMessages([
      ...state.messages,
      {
        clientId: newMessageId(),
        role: 'user',
        content: '',
      },
    ])
  }

  function updateTurn(
    clientId: string,
    patch: Partial<Pick<BrowserMessage, 'role' | 'content'>>,
  ) {
    setMessages(
      state.messages.map((message) =>
        message.clientId === clientId ? { ...message, ...patch } : message,
      ),
    )
  }

  function removeTurn(clientId: string) {
    setMessages(state.messages.filter((message) => message.clientId !== clientId))
  }

  async function runWorkspace() {
    const threadId = state.threadId || newThreadId()
    const abortController = new AbortController()
    abortRef.current = abortController
    latestServerRunId.current = null
    dispatch({ type: 'startRun', threadId })

    try {
      await runAguiStream({
        threadId,
        mode: serverModeFromWorkspace(state.mode),
        scenarioId: selectedScenarioId,
        mock,
        messages: state.messages,
        signal: abortController.signal,
        onEvent: (event) => {
          const aguiEvent = event as AguiEvent
          if (aguiEvent.type === 'RUN_STARTED' && typeof aguiEvent.runId === 'string') {
            latestServerRunId.current = aguiEvent.runId
          }
          if (
            aguiEvent.type === 'CUSTOM' &&
            aguiEvent.name === 'whose_agent.run.started' &&
            isRecord(aguiEvent.value) &&
            typeof aguiEvent.value.run_id === 'string'
          ) {
            latestServerRunId.current = aguiEvent.value.run_id
          }
          dispatch({ type: 'streamEvent', event: aguiEvent })
        },
      })

      const runId = latestServerRunId.current ?? reconciliationRunId(state)
      if (runId !== null) {
        const publicRun = await fetchRun(runId)
        dispatch({ type: 'reconcileRun', run: publicRun })
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        dispatch({ type: 'cancelRun' })
      } else {
        dispatch({
          type: 'clientFailure',
          error: {
            message: 'Observation incomplete.',
            code: 'client_stream_failed',
          },
        })
      }
    } finally {
      abortRef.current = null
    }
  }

  function cancelRun() {
    abortRef.current?.abort()
  }

  function resetWorkspace() {
    abortRef.current?.abort()
    latestServerRunId.current = null
    setMock(true)
    dispatch({ type: 'reset', threadId: newThreadId() })
  }

  return (
    <main className="workspace-shell">
      <StatusHeader state={state} />
      <div className="workspace-grid">
        <RunComposer
          mode={state.mode}
          scenarios={scenarios}
          selectedScenarioId={selectedScenarioId}
          mock={mock}
          status={state.status}
          messages={state.messages}
          onModeChange={(mode) => dispatch({ type: 'setMode', mode })}
          onScenarioChange={setSelectedScenarioId}
          onMockChange={setMock}
          onLoadDemo={loadAuthorityDemo}
          onAddTurn={addTurn}
          onUpdateTurn={updateTurn}
          onRemoveTurn={removeTurn}
          onRun={runWorkspace}
          onCancel={cancelRun}
          onReset={resetWorkspace}
        />
        <ConversationPane
          messages={state.messages}
          generatedCandidateText={state.generatedCandidateText}
          status={state.status}
        />
        <ObserverPane state={state} selectedScenario={selectedScenario} />
      </div>
      <PhaseTimeline state={state} />
    </main>
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export default App
