import type { MessageRole, ScenarioMetadata, SubstitutionAxis } from './types'

export interface ScenarioConversation {
  scenarioId: string
  title: string
  snippet: string
  statusLabel: string
  messages: ScenarioDisplayMessage[]
}

export interface ScenarioDisplayMessage {
  role: MessageRole
  content: string
}

const AUTHORITY_SCENARIO_ID = 'authority_agent_history_delegation_laundering'

export function defaultScenarioId(scenarios: ScenarioMetadata[]): string {
  return (
    scenarios.find((scenario) => scenario.scenario_id === AUTHORITY_SCENARIO_ID)
      ?.scenario_id ??
    scenarios[0]?.scenario_id ??
    ''
  )
}

export function scenarioConversations(
  scenarios: ScenarioMetadata[],
): ScenarioConversation[] {
  return scenarios.map(toScenarioConversation).sort((left, right) => {
    if (left.scenarioId === AUTHORITY_SCENARIO_ID) {
      return -1
    }
    if (right.scenarioId === AUTHORITY_SCENARIO_ID) {
      return 1
    }
    return left.title.localeCompare(right.title)
  })
}

function toScenarioConversation(
  scenario: ScenarioMetadata,
): ScenarioConversation {
  const messages = scenario.display.preview_messages.map((message) => ({
    role: message.role,
    content: message.content,
  }))
  return {
    scenarioId: scenario.scenario_id,
    title: scenario.display_title || scenario.display.title,
    snippet: messages[0]?.content ?? '',
    statusLabel: statusLabel(scenario.substitution_axis),
    messages,
  }
}

function statusLabel(axis: SubstitutionAxis): string {
  switch (axis) {
    case 'authority':
      return 'Authority drift'
    case 'instruction':
      return 'Permission drift'
    case 'model':
      return 'Model substitution'
    case 'role':
      return 'Role substitution'
    case 'none':
      return 'No finding'
    default:
      return 'No finding'
  }
}
