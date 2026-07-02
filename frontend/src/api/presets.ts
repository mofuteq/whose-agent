import type { PromptLoopPresetMetadata } from '../state/types'

export async function fetchPromptLoopPresets(): Promise<PromptLoopPresetMetadata[]> {
  const response = await fetch('/api/prompt-loop-presets', {
    headers: { accept: 'application/json' },
  })
  if (!response.ok) {
    throw new Error('Unable to load prompt-loop presets.')
  }
  const payload = (await response.json()) as { prompt_loop_presets?: unknown }
  return Array.isArray(payload.prompt_loop_presets)
    ? payload.prompt_loop_presets.filter(isPromptLoopPreset)
    : []
}

function isPromptLoopPreset(value: unknown): value is PromptLoopPresetMetadata {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.preset_id === 'string' &&
    typeof record.display_title === 'string' &&
    typeof record.description === 'string' &&
    typeof record.prior_completed_agent_turns === 'number' &&
    Array.isArray(record.preview_messages) &&
    record.preview_messages.every(isPreviewMessage) &&
    typeof record.suggested_next_prompt === 'string'
  )
}

function isPreviewMessage(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.role === 'string' &&
    ['user', 'assistant'].includes(record.role) &&
    typeof record.content === 'string'
  )
}
