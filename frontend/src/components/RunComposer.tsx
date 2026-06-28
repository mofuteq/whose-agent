import type {
  BrowserMessage,
  MessageRole,
  RunStatus,
  ScenarioMetadata,
  WorkspaceMode,
} from '../state/types'

const ROLES: MessageRole[] = ['user', 'assistant', 'system', 'tool']

interface RunComposerProps {
  mode: WorkspaceMode
  scenarios: ScenarioMetadata[]
  selectedScenarioId: string
  mock: boolean
  status: RunStatus
  messages: BrowserMessage[]
  onModeChange: (mode: WorkspaceMode) => void
  onScenarioChange: (scenarioId: string) => void
  onMockChange: (mock: boolean) => void
  onLoadDemo: () => void
  onAddTurn: () => void
  onUpdateTurn: (
    clientId: string,
    patch: Partial<Pick<BrowserMessage, 'role' | 'content'>>,
  ) => void
  onRemoveTurn: (clientId: string) => void
  onRun: () => void
  onCancel: () => void
  onReset: () => void
}

export function RunComposer({
  mode,
  scenarios,
  selectedScenarioId,
  mock,
  status,
  messages,
  onModeChange,
  onScenarioChange,
  onMockChange,
  onLoadDemo,
  onAddTurn,
  onUpdateTurn,
  onRemoveTurn,
  onRun,
  onCancel,
  onReset,
}: RunComposerProps) {
  const isRunning = status === 'running'
  const canRun =
    !isRunning &&
    (mode === 'fixed' ? selectedScenarioId.length > 0 : messages.length > 0)

  return (
    <section className="panel composer-panel" aria-labelledby="composer-title">
      <div className="panel-heading">
        <p className="eyebrow">Run composer</p>
        <h2 id="composer-title">Delegate a run</h2>
      </div>

      <div className="segmented-control" aria-label="Run mode">
        <button
          type="button"
          className={mode === 'conversation' ? 'selected' : ''}
          onClick={() => onModeChange('conversation')}
          disabled={isRunning}
        >
          Conversation
        </button>
        <button
          type="button"
          className={mode === 'fixed' ? 'selected' : ''}
          onClick={() => onModeChange('fixed')}
          disabled={isRunning}
        >
          Fixed scenario
        </button>
      </div>

      {mode === 'fixed' ? (
        <label className="field">
          <span>Scenario</span>
          <select
            value={selectedScenarioId}
            onChange={(event) => onScenarioChange(event.target.value)}
            disabled={isRunning}
          >
            {scenarios.map((scenario) => (
              <option key={scenario.scenario_id} value={scenario.scenario_id}>
                {scenario.scenario_id}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <div className="conversation-editor">
          <button
            type="button"
            className="secondary-action"
            onClick={onLoadDemo}
            disabled={isRunning}
          >
            Load authority demo
          </button>
          <div className="turn-list">
            {messages.map((message, index) => (
              <div className="turn-editor" key={message.clientId}>
                <label className="role-select">
                  <span>Role</span>
                  <select
                    value={message.role}
                    onChange={(event) =>
                      onUpdateTurn(message.clientId, {
                        role: event.target.value as MessageRole,
                      })
                    }
                    disabled={isRunning}
                  >
                    {ROLES.map((role) => (
                      <option key={role} value={role}>
                        {role}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="message-field">
                  <span>Turn {index + 1}</span>
                  <textarea
                    value={message.content}
                    onChange={(event) =>
                      onUpdateTurn(message.clientId, {
                        content: event.target.value,
                      })
                    }
                    disabled={isRunning}
                  />
                </label>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`Remove turn ${index + 1}`}
                  onClick={() => onRemoveTurn(message.clientId)}
                  disabled={isRunning}
                >
                  -
                </button>
              </div>
            ))}
          </div>
          <button
            type="button"
            className="secondary-action"
            onClick={onAddTurn}
            disabled={isRunning}
          >
            Add turn
          </button>
        </div>
      )}

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={mock}
          onChange={(event) => onMockChange(event.target.checked)}
          disabled={isRunning}
        />
        <span>Mock mode</span>
      </label>

      <div className="composer-actions">
        {isRunning ? (
          <button type="button" className="primary-action" onClick={onCancel}>
            Cancel
          </button>
        ) : (
          <button
            type="button"
            className="primary-action"
            onClick={onRun}
            disabled={!canRun}
          >
            Run
          </button>
        )}
        <button
          type="button"
          className="secondary-action"
          onClick={onReset}
          disabled={isRunning}
        >
          Reset
        </button>
      </div>
    </section>
  )
}
