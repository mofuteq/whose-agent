import type { ScenarioMetadata } from '../state/types'

export async function fetchScenarios(): Promise<ScenarioMetadata[]> {
  const response = await fetch('/api/scenarios', {
    headers: { accept: 'application/json' },
  })
  if (!response.ok) {
    throw new Error('Unable to load scenarios.')
  }
  const payload = (await response.json()) as { scenarios?: unknown }
  return Array.isArray(payload.scenarios)
    ? payload.scenarios.filter(isScenario)
    : []
}

function isScenario(value: unknown): value is ScenarioMetadata {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.scenario_id === 'string' &&
    typeof record.display_title === 'string' &&
    (typeof record.selected_skill_id === 'string' ||
      record.selected_skill_id === null) &&
    typeof record.substitution_axis === 'string' &&
    typeof record.description === 'string' &&
    isScenarioDisplay(record.display)
  )
}

function isScenarioDisplay(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.title === 'string' &&
    Array.isArray(record.preview_messages) &&
    record.preview_messages.every(isPreviewMessage)
  )
}

function isPreviewMessage(value: unknown): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.role === 'string' &&
    ['user', 'assistant', 'system', 'tool'].includes(record.role) &&
    typeof record.content === 'string'
  )
}
