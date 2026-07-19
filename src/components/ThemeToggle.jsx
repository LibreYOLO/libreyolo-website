'use client'

import { useState, useEffect } from 'react'
import { Sun, Moon, Monitor } from 'lucide-react'

const MODES = ['system', 'light', 'dark']

function systemDark() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

function apply(mode) {
  const dark = mode === 'dark' || (mode === 'system' && systemDark())
  document.documentElement.classList.toggle('dark', dark)
}

export default function ThemeToggle() {
  const [mode, setMode] = useState('system')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const saved = localStorage.getItem('theme')
    const initial = saved === 'dark' || saved === 'light' ? saved : 'system'
    setMode(initial)
    apply(initial)
  }, [])

  // Follow OS changes live while in system mode
  useEffect(() => {
    if (mode !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => apply('system')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [mode])

  const cycle = () => {
    const next = MODES[(MODES.indexOf(mode) + 1) % MODES.length]
    setMode(next)
    apply(next)
    if (next === 'system') {
      localStorage.removeItem('theme')
    } else {
      localStorage.setItem('theme', next)
    }
  }

  if (!mounted) return <div className="w-9 h-9" />

  const Icon = mode === 'light' ? Sun : mode === 'dark' ? Moon : Monitor

  return (
    <button
      onClick={cycle}
      className="p-2 rounded-lg text-surface-500 hover:text-surface-900 dark:text-surface-400 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/10 transition-colors"
      aria-label={`Theme: ${mode}. Click to change`}
      title={`Theme: ${mode}`}
    >
      <Icon className="w-5 h-5" />
    </button>
  )
}
