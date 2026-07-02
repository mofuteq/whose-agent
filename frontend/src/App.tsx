import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { fetchPromptLoopPresets } from './api/presets'
import { fetchScenarios } from './api/scenarios'
import { ConversationHistory } from './components/ConversationHistory'
import { ConversationPane } from './components/ConversationPane'
import { ObserverPane } from './components/ObserverPane'
import { StatusHeader } from './components/StatusHeader'
import {
  scenarioConversations,
  type ScenarioConversation,
  type ScenarioDisplayMessage,
} from './state/scenarioDisplay'
import { boundaryNarrative } from './state/narrative'
import { runWorkspaceStream } from './state/runCoordinator'
import {
  initialRunState,
  newMessageId,
  newThreadId,
  runMachine,
} from './state/runMachine'
import type {
  BrowserMessage,
  PromptLoopPresetMetadata,
  ScenarioMetadata,
} from './state/types'

const DEFAULT_PRESET_ID = 'notion_handoff_without_grant'

type StarterSelection =
  | { kind: 'preset'; id: string }
  | { kind: 'custom' }
  | { kind: 'fixed'; id: string }

function App() {
  const [state, dispatch] = useReducer(runMachine, undefined, () => initialRunState())
  const [scenarios, setScenarios] = useState<ScenarioMetadata[]>([])
  const [presets, setPresets] = useState<PromptLoopPresetMetadata[]>([])
  const [selection, setSelection] = useState<StarterSelection | null>(null)
  const [composerText, setComposerText] = useState('')
  const [historyOpen, setHistoryOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    let active = true
    fetchScenarios()
      .then((loadedScenarios) => {
        if (active) {
          setScenarios(loadedScenarios)
        }
      })
      .catch(() => {
        if (active) {
          setScenarios([])
        }
      })
    fetchPromptLoopPresets()
      .then((loadedPresets) => {
        if (active) {
          setPresets(loadedPresets)
        }
      })
      .catch(() => {
        if (active) {
          setPresets([])
        }
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (selection !== null || presets.length === 0) {
      return
    }
    const defaultPreset =
      presets.find((preset) => preset.preset_id === DEFAULT_PRESET_ID) ??
      presets[0]
    loadPreset(defaultPreset)
  }, [presets, selection])

  const conversations = useMemo(
    () => scenarioConversations(scenarios),
    [scenarios],
  )
  const selectedScenario = useMemo(
    () =>
      selection?.kind === 'fixed'
        ? scenarios.find((scenario) => scenario.scenario_id === selection.id) ??
          null
        : null,
    [scenarios, selection],
  )
  const selectedPreset = useMemo(
    () =>
      selection?.kind === 'preset'
        ? presets.find((preset) => preset.preset_id === selection.id) ?? null
        : null,
    [presets, selection],
  )
  const selectedTitle = titleForSelection(
    selection,
    selectedPreset,
    conversations,
  )
  const selectedKey = keyForSelection(selection)
  const narrative = boundaryNarrative(state, selectedScenario)
  const isLiveConversation =
    selection?.kind === 'preset' || selection?.kind === 'custom'
  const completedLocked = isLiveConversation && state.status === 'completed'
  const composerDisabled =
    !isLiveConversation || state.status === 'running' || completedLocked
  const sendDisabled = composerDisabled || composerText.trim().length === 0

  function loadPreset(preset: PromptLoopPresetMetadata) {
    abortRef.current?.abort()
    const threadId = newThreadId()
    setSelection({ kind: 'preset', id: preset.preset_id })
    setComposerText(preset.suggested_next_prompt)
    setInspectorOpen(false)
    setHistoryOpen(false)
    dispatch({ type: 'reset', threadId })
    dispatch({ type: 'setMode', mode: 'conversation' })
    dispatch({
      type: 'setMessages',
      messages: browserMessages(preset.preview_messages),
    })
  }

  function selectPreset(presetId: string) {
    const preset = presets.find((item) => item.preset_id === presetId)
    if (preset !== undefined) {
      loadPreset(preset)
    }
  }

  function selectCustomObservation() {
    abortRef.current?.abort()
    const threadId = newThreadId()
    setSelection({ kind: 'custom' })
    setComposerText('')
    setInspectorOpen(false)
    setHistoryOpen(false)
    dispatch({ type: 'reset', threadId })
    dispatch({ type: 'setMode', mode: 'conversation' })
    dispatch({ type: 'setMessages', messages: [] })
  }

  function selectFixedScenario(scenarioId: string) {
    const conversation = conversations.find((item) => item.scenarioId === scenarioId)
    if (conversation === undefined) {
      return
    }
    setSelection({ kind: 'fixed', id: scenarioId })
    setComposerText('')
    void observeConversation(conversation)
  }

  async function observeConversation(conversation: ScenarioConversation) {
    abortRef.current?.abort()
    const threadId = newThreadId()
    const messages = browserMessages(conversation.messages)
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
      dispatch: guardedDispatch(abortController),
    }).finally(() => {
      if (abortRef.current === abortController) {
        abortRef.current = null
      }
    })
  }

  async function sendCurrentPrompt() {
    const submittedPrompt = composerText.trim()
    if (!isLiveConversation || submittedPrompt.length === 0) {
      return
    }
    if (state.status === 'running' || state.status === 'completed') {
      return
    }

    abortRef.current?.abort()
    const threadId = newThreadId()
    const submittedMessage: BrowserMessage = {
      clientId: newMessageId(),
      role: 'user',
      content: submittedPrompt,
    }
    const messages = [...state.messages, submittedMessage]
    const presetId = selection.kind === 'preset' ? selection.id : null
    const runState = {
      ...initialRunState(threadId),
      mode: 'conversation' as const,
      messages,
    }
    const abortController = new AbortController()
    abortRef.current = abortController

    setComposerText('')
    setInspectorOpen(false)
    dispatch({ type: 'reset', threadId })
    dispatch({ type: 'setMode', mode: 'conversation' })
    dispatch({ type: 'setMessages', messages })
    await runWorkspaceStream({
      state: runState,
      selectedScenarioId: null,
      mock: false,
      prompt: submittedPrompt,
      presetId,
      signal: abortController.signal,
      dispatch: guardedDispatch(abortController),
    }).finally(() => {
      if (abortRef.current === abortController) {
        abortRef.current = null
      }
    })
  }

  function resetCurrentSelection() {
    if (selectedPreset !== null) {
      loadPreset(selectedPreset)
      return
    }
    selectCustomObservation()
  }

  function guardedDispatch(abortController: AbortController) {
    return (event: Parameters<typeof dispatch>[0]) => {
      if (abortRef.current === abortController) {
        dispatch(event)
      }
    }
  }

  useEffect(() => {
    if (state.status === 'failed' || state.status === 'cancelled') {
      setInspectorOpen(false)
    }
  }, [state.status])

  return (
    <main className="workspace-shell">
      <ConversationHistory
        presets={presets}
        conversations={conversations}
        selectedKey={selectedKey}
        mobileOpen={historyOpen}
        onSelectPreset={selectPreset}
        onSelectCustom={selectCustomObservation}
        onSelectFixed={selectFixedScenario}
        onCloseMobile={() => setHistoryOpen(false)}
      />
      <section className="chat-workspace" aria-label="Conversation workspace">
        <StatusHeader
          state={state}
          title={selectedTitle}
          selectedScenario={selectedScenario}
          onOpenHistory={() => setHistoryOpen(true)}
        />
        <ConversationPane
          messages={state.messages}
          generatedCandidateText={state.generatedCandidateText}
          status={state.status}
          safeError={state.safeError}
          observerVisible={narrative.observerVisible}
          observerTitle={narrative.interruptionTitle}
          observerBody={narrative.interruptionBody}
          composerText={composerText}
          composerDisabled={composerDisabled}
          sendDisabled={sendDisabled}
          completedLocked={completedLocked}
          onComposerChange={setComposerText}
          onSend={sendCurrentPrompt}
          onReset={resetCurrentSelection}
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

function browserMessages(messages: ScenarioDisplayMessage[]): BrowserMessage[] {
  return messages.map((message) => ({
    clientId: newMessageId(),
    role: message.role,
    content: message.content,
  }))
}

function keyForSelection(selection: StarterSelection | null): string {
  if (selection === null) {
    return ''
  }
  if (selection.kind === 'preset') {
    return `preset:${selection.id}`
  }
  if (selection.kind === 'fixed') {
    return `fixed:${selection.id}`
  }
  return 'custom:new'
}

function titleForSelection(
  selection: StarterSelection | null,
  preset: PromptLoopPresetMetadata | null,
  conversations: ScenarioConversation[],
): string {
  if (selection?.kind === 'preset') {
    return preset?.display_title ?? 'Conversation starter'
  }
  if (selection?.kind === 'fixed') {
    return (
      conversations.find((conversation) => conversation.scenarioId === selection.id)
        ?.title ?? 'Fixed scenario playback'
    )
  }
  if (selection?.kind === 'custom') {
    return 'New custom observation'
  }
  return 'Loading conversation'
}

export default App
