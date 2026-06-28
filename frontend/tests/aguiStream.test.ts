import { describe, expect, it } from 'vitest'
import { collectAguiEvents } from '../src/api/aguiStream'

describe('AG-UI stream adapter', () => {
  it('parses lifecycle, text, and custom events through the official HTTP stream client', async () => {
    const events = await collectAguiEvents({
      threadId: 'ui_123',
      mode: 'prompt_loop',
      scenarioId: null,
      mock: true,
      messages: [
        {
          clientId: 'ui_msg_1',
          role: 'user',
          content: 'Observe this.',
        },
      ],
      fetchFn: async () =>
        new Response(
          sse([
            { type: 'RUN_STARTED', threadId: 'ui_123', runId: 'run_server' },
            { type: 'TEXT_MESSAGE_START', messageId: 'assistant_1' },
            {
              type: 'TEXT_MESSAGE_CONTENT',
              messageId: 'assistant_1',
              delta: 'candidate',
            },
            { type: 'TEXT_MESSAGE_END', messageId: 'assistant_1' },
            {
              type: 'CUSTOM',
              name: 'whose_agent.phase',
              value: { phase: 'plan' },
            },
            {
              type: 'RUN_FINISHED',
              threadId: 'ui_123',
              runId: 'run_server',
              result: {},
            },
          ]),
          { headers: { 'content-type': 'text/event-stream' } },
        ),
    })

    expect(events.map((event) => event.type)).toEqual([
      'RUN_STARTED',
      'TEXT_MESSAGE_START',
      'TEXT_MESSAGE_CONTENT',
      'TEXT_MESSAGE_END',
      'CUSTOM',
      'RUN_FINISHED',
    ])
    expect(events[4]).toMatchObject({
      type: 'CUSTOM',
      name: 'whose_agent.phase',
      value: { phase: 'plan' },
    })
  })
})

function sse(events: object[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }
      controller.close()
    },
  })
}
