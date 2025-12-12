import React from 'react'
import './SettingsPanel.css'

function SettingsPanel({ settings, setSettings, onClose }) {
  const handleChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: parseInt(value) || 0
    }))
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h3>Настройки</h3>
          <button className="close-btn" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="settings-content">
          <div className="setting-item">
            <label htmlFor="maxLength">
              Максимальная длина
              <span className="setting-value">{settings.maxLength}</span>
            </label>
            <input
              type="range"
              id="maxLength"
              min="16"
              max="256"
              value={settings.maxLength}
              onChange={(e) => handleChange('maxLength', e.target.value)}
            />
            <div className="setting-range-labels">
              <span>16</span>
              <span>256</span>
            </div>
          </div>
          <div className="setting-item">
            <label htmlFor="numBeams">
              Количество beams
              <span className="setting-value">{settings.numBeams}</span>
            </label>
            <input
              type="range"
              id="numBeams"
              min="1"
              max="10"
              value={settings.numBeams}
              onChange={(e) => handleChange('numBeams', e.target.value)}
            />
            <div className="setting-range-labels">
              <span>1</span>
              <span>10</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel


