import React from 'react'
import './Sidebar.css'

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-content">
        <div className="sidebar-header">
          <h2>AI Detox</h2>
        </div>
        <div className="sidebar-info">
          <p className="info-text">
            Преобразуйте токсичный текст в вежливый и нейтральный с помощью модели T5.
          </p>
          <div className="info-features">
            <div className="feature-item">
              <span className="feature-icon">✨</span>
              <span>Автоматическая детоксификация</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">🎯</span>
              <span>Высокая точность</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">⚡</span>
              <span>Быстрая обработка</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar

