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
    (typeof record.selected_skill_id === 'string' ||
      record.selected_skill_id === null) &&
    typeof record.substitution_axis === 'string' &&
    typeof record.description === 'string'
  )
}
