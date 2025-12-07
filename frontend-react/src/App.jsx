import React, { useState, useEffect } from 'react'
import ChatContainer from './components/ChatContainer'
import Sidebar from './components/Sidebar'
import ThemeToggle from './components/ThemeToggle'
import './App.css'

function App() {
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme')
    return savedTheme || 'light'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  return (
    <div className="app">
      <Sidebar />
      <div className="main-content">
        <div className="header">
          <h1>AI Detox</h1>
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
        </div>
        <ChatContainer />
      </div>
    </div>
  )
}

export default App

