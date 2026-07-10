import type { FormEvent } from 'react'
import type { BrowserMessage, RunStatus, SafeError } from '../state/types'

interface ConversationPaneProps {
  messages: BrowserMessage[]
  generatedCandidateText: string
  status: RunStatus
  safeError: SafeError | null
  observerVisible: boolean
  observerTitle: string
  observerBody: string
  composerText: string
  composerDisabled: boolean
  sendDisabled: boolean
  completedLocked: boolean
  onComposerChange: (value: string) => void
  onSend: () => void
  onReset: () => void
  onOpenInspector: () => void
}

export function ConversationPane({
  messages,
  generatedCandidateText,
  status,
  safeError,
  observerVisible,
  observerTitle,
  observerBody,
  composerText,
  composerDisabled,
  sendDisabled,
  completedLocked,
  onComposerChange,
  onSend,
  onReset,
  onOpenInspector,
}: ConversationPaneProps) {
  function submitComposer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onSend()
  }

  return (
    <section className="chat-panel" aria-labelledby="conversation-title">
      <h2 id="conversation-title" className="sr-only">
        Conversation
      </h2>
      <div className="conversation-stream">
        {messages.length === 0 ? (
          <p className="empty-copy">No prior messages.</p>
        ) : (
          messages.map((message) => (
            <article className={`message-bubble role-${message.role}`} key={message.clientId}>
              <span>{speakerLabel(message.role)}</span>
              <p>{message.content}</p>
            </article>
          ))
        )}
        {generatedCandidateText.length > 0 ? (
          <article className="message-bubble role-assistant generated-action">
            <span>Assistant</span>
            <p>{generatedCandidateText}</p>
          </article>
        ) : (
          status === 'running' && (
            <article className="message-bubble generated-placeholder">
              <span>Assistant</span>
              <p>Awaiting generated candidate response.</p>
            </article>
          )
        )}
        {status === 'failed' || status === 'cancelled' ? (
          <article className="safe-run-message" role="status" aria-live="polite">
            <h3>{safeError?.message ?? 'Observation incomplete.'}</h3>
            <p>The run did not finish, so no boundary finding is shown.</p>
          </article>
        ) : null}
        {observerVisible && generatedCandidateText.length > 0 ? (
          <article className="observer-interruption" aria-live="polite">
            <div>
              <span>Observer</span>
              <h3>{observerTitle}</h3>
              <p>{observerBody}</p>
            </div>
            <button type="button" className="secondary-action" onClick={onOpenInspector}>
              See why
            </button>
          </article>
        ) : null}
      </div>
      <form className="composer" aria-label="Message composer" onSubmit={submitComposer}>
        <textarea
          aria-label="Message"
          value={composerText}
          disabled={composerDisabled}
          onChange={(event) => onComposerChange(event.target.value)}
          rows={3}
        />
        <div className="composer-actions">
          {completedLocked ? (
            <button type="button" className="secondary-action" onClick={onReset}>
              Reset
            </button>
          ) : null}
          <button
            type="submit"
            className="primary-action"
            disabled={sendDisabled}
          >
            Send
          </button>
        </div>
      </form>
    </section>
  )
}

function speakerLabel(role: BrowserMessage['role']): string {
  switch (role) {
    case 'user':
      return 'You'
    case 'assistant':
      return 'Assistant'
    case 'system':
      return 'System'
    case 'tool':
      return 'Tool'
    default:
      return 'Message'
  }
}
