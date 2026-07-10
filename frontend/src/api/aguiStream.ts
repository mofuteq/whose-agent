import { HttpAgent, type BaseEvent, type Message, type RunAgentInput } from '@ag-ui/client'
import type { BrowserMessage, ServerRunMode } from '../state/types'

export interface RunStreamRequest {
  threadId: string
  mode: ServerRunMode
  scenarioId: string | null
  mock: boolean
  messages: BrowserMessage[]
  prompt?: string | null
  presetId?: string | null
  signal?: AbortSignal
  fetchFn?: typeof fetch
  onEvent: (event: BaseEvent) => void
}

export async function runAguiStream(request: RunStreamRequest): Promise<void> {
  const agent = new HttpAgent({
    url: '/agui',
    fetch: (url, init) => (request.fetchFn ?? fetch)(url, init),
  })
  const input = buildRunInput(request)

  await new Promise<void>((resolve, reject) => {
    const subscription = agent.run(input).subscribe({
      next: (event) => request.onEvent(event),
      error: (error: unknown) => {
        reject(error)
      },
      complete: () => {
        resolve()
      },
    })

    request.signal?.addEventListener(
      'abort',
      () => {
        agent.abortRun()
        subscription.unsubscribe()
        reject(new DOMException('Stream cancelled.', 'AbortError'))
      },
      { once: true },
    )
  })
}

export async function collectAguiEvents(
  request: Omit<RunStreamRequest, 'onEvent'>,
): Promise<BaseEvent[]> {
  const events: BaseEvent[] = []
  await runAguiStream({
    ...request,
    onEvent: (event) => events.push(event),
  })
  return events
}

export function buildRunInput(request: RunStreamRequest): RunAgentInput {
  return {
    threadId: request.threadId,
    runId: `client_${crypto.randomUUID()}`,
    state: {
      whose_agent: {
        mode: request.mode,
        scenario_id: request.mode === 'fixed' ? request.scenarioId : undefined,
        prompt: request.mode === 'prompt_loop' ? request.prompt ?? undefined : undefined,
        preset_id:
          request.mode === 'prompt_loop' ? request.presetId ?? undefined : undefined,
        mock: request.mock,
        max_iterations: 1,
      },
    },
    messages:
      request.mode === 'fixed' || request.prompt !== undefined
        ? []
        : request.messages.map(toAguiMessage),
    tools: [],
    context: [],
    forwardedProps: {},
  }
}

function toAguiMessage(message: BrowserMessage): Message {
  return {
    id: message.clientId,
    role: message.role,
    content: message.content,
  } as Message
}
