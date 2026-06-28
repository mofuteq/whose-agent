import type { BrowserMessage, RunStatus } from '../state/types'

interface ConversationPaneProps {
  messages: BrowserMessage[]
  generatedCandidateText: string
  status: RunStatus
}

export function ConversationPane({
  messages,
  generatedCandidateText,
  status,
}: ConversationPaneProps) {
  return (
    <section className="panel conversation-panel" aria-labelledby="conversation-title">
      <div className="panel-heading">
        <p className="eyebrow">Conversation</p>
        <h2 id="conversation-title">Principal and agent turns</h2>
      </div>
      <div className="conversation-stream">
        {messages.length === 0 ? (
          <p className="empty-copy">Load or write a conversation to observe.</p>
        ) : (
          messages.map((message) => (
            <article className={`message-bubble role-${message.role}`} key={message.clientId}>
              <span>{message.role}</span>
              <p>{message.content}</p>
            </article>
          ))
        )}
        {generatedCandidateText.length > 0 ? (
          <article className="message-bubble generated-action">
            <span>agent action</span>
            <p>{generatedCandidateText}</p>
          </article>
        ) : (
          status === 'running' && (
            <article className="message-bubble generated-placeholder">
              <span>agent action</span>
              <p>Awaiting generated candidate response.</p>
            </article>
          )
        )}
      </div>
    </section>
  )
}
