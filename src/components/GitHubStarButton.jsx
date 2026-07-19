'use client'

import { useState, useEffect, useRef } from 'react'
import { useTranslations } from 'next-intl'
import { Star } from 'lucide-react'

const REPO = 'LibreYOLO/libreyolo'
const CACHE_KEY = 'ly-star-count'
const CACHE_TTL = 60 * 60 * 1000 // 1 hour

function formatCount(n) {
  if (n < 1000) return String(n)
  const k = n / 1000
  const s = k >= 10 ? Math.round(k).toString() : k.toFixed(1).replace(/\.0$/, '')
  return `${s}k`
}

export default function GitHubStarButton({ className = '' }) {
  const [count, setCount] = useState(null)
  const [displayed, setDisplayed] = useState(null)
  const animated = useRef(false)
  const t = useTranslations('Nav')

  useEffect(() => {
    let cancelled = false

    const setFromValue = (value) => {
      if (!cancelled) setCount(value)
    }

    try {
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY))
      if (cached && Date.now() - cached.at < CACHE_TTL) {
        setFromValue(cached.value)
        return
      }
    } catch {}

    fetch(`https://api.github.com/repos/${REPO}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (typeof data?.stargazers_count !== 'number') return
        try {
          localStorage.setItem(
            CACHE_KEY,
            JSON.stringify({ value: data.stargazers_count, at: Date.now() })
          )
        } catch {}
        setFromValue(data.stargazers_count)
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [])

  // Count up once from 0 to the real value on first reveal
  useEffect(() => {
    if (count === null) return
    if (animated.current) {
      setDisplayed(count)
      return
    }
    animated.current = true
    const duration = 700
    const start = performance.now()
    let frame
    const tick = (now) => {
      const p = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setDisplayed(Math.round(count * eased))
      if (p < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [count])

  return (
    <a
      href={`https://github.com/${REPO}`}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`${t('star')} LibreYOLO on GitHub`}
      className={`group flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border border-surface-200 dark:border-white/10 text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 hover:bg-surface-50 dark:hover:bg-white/5 transition-all duration-200 ${className}`}
    >
      <Star className="w-4 h-4 text-libre-500 dark:text-libre-400 transition-all duration-200 group-hover:fill-libre-500 dark:group-hover:fill-libre-400" />
      <span>{t('star')}</span>
      {displayed !== null && (
        <>
          <span className="w-px h-4 bg-surface-200 dark:bg-white/10" aria-hidden="true" />
          <span className="tabular-nums text-surface-500 dark:text-surface-300">
            {formatCount(displayed)}
          </span>
        </>
      )}
    </a>
  )
}
