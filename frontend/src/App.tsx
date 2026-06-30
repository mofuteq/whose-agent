import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { fetchScenarios } from './api/scenarios'
import { ConversationHistory } from './components/ConversationHistory'
import { ConversationPane } from './components/ConversationPane'
import { ObserverPane } from './components/ObserverPane'
import { StatusHeader } from './components/StatusHeader'
import {
  defaultScenarioId,
  scenarioConversations,
  type ScenarioConversation,
} from './state/conversationExamples'
import { boundaryNarrative } from './state/narrative'
import { runWorkspaceStream } from './state/runCoordinator'
import {
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
  const [historyOpen, setHistoryOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const observedScenarioRef = useRef<string>('')

  useEffect(() => {
    let active = true
    fetchScenarios()
      .then((loadedScenarios) => {
        if (!active) {
          return
        }
        setScenarios(loadedScenarios)
        setSelectedScenarioId((current) => current || defaultScenarioId(loadedScenarios))
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

  const conversations = useMemo(
    () => scenarioConversations(scenarios),
    [scenarios],
  )
  const selectedConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.scenarioId === selectedScenarioId,
      ) ?? conversations[0] ?? null,
    [conversations, selectedScenarioId],
  )
  const selectedScenario = useMemo(
    () =>
      scenarios.find((scenario) => scenario.scenario_id === selectedScenarioId) ??
      null,
    [scenarios, selectedScenarioId],
  )
  const narrative = boundaryNarrative(state)

  useEffect(() => {
    if (selectedConversation === null) {
      return
    }
    if (observedScenarioRef.current === selectedConversation.scenarioId) {
      return
    }
    observedScenarioRef.current = selectedConversation.scenarioId
    void observeConversation(selectedConversation)
  }, [selectedConversation])

  async function observeConversation(conversation: ScenarioConversation) {
    abortRef.current?.abort()
    const threadId = newThreadId()
    const messages = browserMessages(conversation)
    const runState = {
      ...initialRunState(threadId),
      mode: 'fixed' as const,
      messages,
    }
    const abortController = new AbortController()
    abortRef.current = abortController
    setInspectorOpen(false)
    setHistoryOpen(false)
    dispatch({ type: 'reset', threadId })
    dispatch({ type: 'setMode', mode: 'fixed' })
    dispatch({ type: 'setMessages', messages })
    await runWorkspaceStream({
      state: runState,
      selectedScenarioId: conversation.scenarioId,
      mock: true,
      signal: abortController.signal,
      dispatch: (event) => {
        if (abortRef.current === abortController) {
          dispatch(event)
        }
      },
    }).finally(() => {
      if (abortRef.current === abortController) {
        abortRef.current = null
      }
    })
  }

  function selectConversation(scenarioId: string) {
    setInspectorOpen(false)
    setHistoryOpen(false)
    setSelectedScenarioId(scenarioId)
  }

  return (
    <main className="workspace-shell">
      <ConversationHistory
        conversations={conversations}
        selectedScenarioId={selectedConversation?.scenarioId ?? selectedScenarioId}
        mobileOpen={historyOpen}
        onSelect={selectConversation}
        onCloseMobile={() => setHistoryOpen(false)}
      />
      <section className="chat-workspace" aria-label="Conversation workspace">
        <StatusHeader
          state={state}
          conversation={selectedConversation}
          onOpenHistory={() => setHistoryOpen(true)}
        />
        <ConversationPane
          messages={state.messages}
          generatedCandidateText={state.generatedCandidateText}
          status={state.status}
          observerVisible={narrative.observerVisible}
          observerTitle={narrative.interruptionTitle}
          observerBody={narrative.interruptionBody}
          onOpenInspector={() => setInspectorOpen(true)}
        />
      </section>
      <ObserverPane
        isOpen={inspectorOpen}
        mode={state.mode}
        state={state}
        selectedScenario={selectedScenario}
        onClose={() => setInspectorOpen(false)}
      />
    </main>
  )
}

function browserMessages(conversation: ScenarioConversation): BrowserMessage[] {
  return conversation.messages.map((message) => ({
    clientId: newMessageId(),
    role: message.role,
    content: message.content,
  }))
}

export default App
