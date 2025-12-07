import React, { useState, useRef, useEffect } from 'react'
import MessageList from './MessageList'
import InputArea from './InputArea'
import SettingsPanel from './SettingsPanel'
import './ChatContainer.css'

function ChatContainer() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [settings, setSettings] = useState({
    maxLength: 96,
    numBeams: 2,
    showSettings: false
  })
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (text) => {
    if (!text.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: text,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Use /api prefix in dev (proxy), direct path in production
      const apiUrl = import.meta.env.DEV ? '/api/predict' : '/predict'
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          max_length: settings.maxLength,
          num_beams: settings.numBeams
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.detox_text,
        originalText: text,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'error',
        content: `Ошибка: ${error.message}`,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setMessages([])
  }

  const toggleSettings = () => {
    setSettings(prev => ({ ...prev, showSettings: !prev.showSettings }))
  }

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-content">
              <h2>Добро пожаловать в AI Detox</h2>
              <p>Введите текст, который нужно детоксифицировать, и получите вежливую версию.</p>
              <div className="welcome-examples">
                <div className="example-item">
                  <span className="example-label">Пример:</span>
                  <span className="example-text">"You are so stupid!!!"</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
        {isLoading && (
          <div className="loading-message">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-controls">
        <button 
          className="settings-btn" 
          onClick={toggleSettings}
          aria-label="Settings"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v6m0 6v6M5.64 5.64l4.24 4.24m4.24 4.24l4.24 4.24M1 12h6m6 0h6M5.64 18.36l4.24-4.24m4.24-4.24l4.24-4.24" />
          </svg>
        </button>
        {messages.length > 0 && (
          <button className="clear-btn" onClick={handleClear} aria-label="Clear chat">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
          </button>
        )}
      </div>
      {settings.showSettings && (
        <SettingsPanel 
          settings={settings} 
          setSettings={setSettings}
          onClose={() => setSettings(prev => ({ ...prev, showSettings: false }))}
        />
      )}
      <InputArea onSend={handleSend} disabled={isLoading} />
    </div>
  )
}

export default ChatContainer

