import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { fetchScenarios } from './api/scenarios'
import { ConversationPane } from './components/ConversationPane'
import { ObserverPane } from './components/ObserverPane'
import { PhaseTimeline } from './components/PhaseTimeline'
import { RunComposer } from './components/RunComposer'
import { StatusHeader } from './components/StatusHeader'
import { runWorkspaceStream } from './state/runCoordinator'
import {
  authorityDemoMessages,
  initialRunState,
  newMessageId,
  newThreadId,
  runMachine,
} from './state/runMachine'
import type { BrowserMessage, ScenarioMetadata } from './state/types'

function App() {
  const [state, dispatch] = useReducer(runMachine, undefined, () => initialRunState())
  const [scenarios, setScenarios] = useState<ScenarioMetadata[]>([])
  const [selectedScenarioId, setSelectedScenarioId] = useState('')
  const [mock, setMock] = useState(true)
  const abortRef = useRef<AbortController | null>(null)

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
    await runWorkspaceStream({
      state: { ...state, threadId },
      selectedScenarioId,
      mock,
      signal: abortController.signal,
      dispatch,
    }).finally(() => {
      abortRef.current = null
    })
  }

  function cancelRun() {
    abortRef.current?.abort()
  }

  function resetWorkspace() {
    abortRef.current?.abort()
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
        <ObserverPane
          mode={state.mode}
          state={state}
          selectedScenario={selectedScenario}
        />
      </div>
      <PhaseTimeline state={state} />
    </main>
  )
}

export default App
