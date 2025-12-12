import React from 'react'
import './Message.css'

function Message({ message }) {
  if (message.role === 'error') {
    return (
      <div className="message message-error">
        <div className="message-content">
          <p>{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`message message-${message.role}`}>
      <div className="message-avatar">
        {message.role === 'user' ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h6M9 15h6" />
          </svg>
        )}
      </div>
      <div className="message-content">
        <p className="message-text">{message.content}</p>
        {message.originalText && message.role === 'assistant' && (
          <div className="message-comparison">
            <div className="comparison-item">
              <span className="comparison-label">Исходный:</span>
              <span className="comparison-text original">{message.originalText}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Message


