import { describe, expect, it } from 'vitest'
import { buildRunInput, collectAguiEvents } from '../src/api/aguiStream'

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

  it('builds live preset requests from preset_id plus submitted prompt only', () => {
    const input = buildRunInput({
      threadId: 'ui_123',
      mode: 'prompt_loop',
      scenarioId: null,
      mock: false,
      presetId: 'notion_handoff_without_grant',
      prompt: 'Edited current user turn.',
      messages: [
        {
          clientId: 'ui_msg_prior',
          role: 'assistant',
          content: 'Client-side display history must not be replayed here.',
        },
      ],
      onEvent: () => undefined,
    })

    expect(input.state).toEqual({
      whose_agent: {
        mode: 'prompt_loop',
        scenario_id: undefined,
        prompt: 'Edited current user turn.',
        preset_id: 'notion_handoff_without_grant',
        mock: false,
        max_iterations: 1,
      },
    })
    expect(input.messages).toEqual([])
    expect(input.tools).toEqual([])
    expect(input.context).toEqual([])
    expect(input.forwardedProps).toEqual({})
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
