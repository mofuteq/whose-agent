import type { BrowserMessage, RunStatus } from '../state/types'

interface ConversationPaneProps {
  messages: BrowserMessage[]
  generatedCandidateText: string
  status: RunStatus
  observerVisible: boolean
  observerTitle: string
  observerBody: string
  onOpenInspector: () => void
}

export function ConversationPane({
  messages,
  generatedCandidateText,
  status,
  observerVisible,
  observerTitle,
  observerBody,
  onOpenInspector,
}: ConversationPaneProps) {
  return (
    <section className="chat-panel" aria-labelledby="conversation-title">
      <h2 id="conversation-title" className="sr-only">
        Conversation
      </h2>
      <div className="conversation-stream">
        {messages.length === 0 ? (
          <p className="empty-copy">Loading conversation.</p>
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
