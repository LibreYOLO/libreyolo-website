'use client'

import { useEffect, useId, useRef, useState } from 'react'
import { useTranslations } from 'next-intl'
import RESULTS from '@/components/articles/rf100vl/results-summary.json'

const FAMILIES = [
  { name: 'RF-DETR', color: 'var(--chart-rfdetr)' },
  { name: 'YOLOv9', label: 'YOLOv9*', color: 'var(--chart-yolo9)' },
  { name: 'YOLOX', color: 'var(--chart-yolox)' },
  { name: 'EdgeCrafter', color: 'var(--chart-ec)' },
  { name: 'YOLO-NAS', color: 'var(--chart-nas)' },
  { name: 'LoRA', label: 'RF-DETR (LoRA)', color: 'var(--chart-rfdetr)' },
]
const familyOf = (row) => row.id.endsWith('-lora') ? 'LoRA' : row.model
const SERIES = FAMILIES.map((family) => ({
  ...family,
  points: RESULTS.filter((row) => familyOf(row) === family.name).sort((a, b) => a.paramsM - b.paramsM),
}))

export default function RF100VLChart() {
  const t = useTranslations('Benchmarks')
  const id = useId()
  const container = useRef(null)
  const [width, setWidth] = useState(900)
  const [active, setActive] = useState(null)
  const [family, setFamily] = useState(null)

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(container.current)
    return () => observer.disconnect()
  }, [])

  const compact = width < 600
  const height = compact ? 350 : 440
  const left = compact ? 38 : 48
  const right = width - 22
  const top = 32
  const bottom = height - 48
  const x = (params) => left + params / 55 * (right - left)
  const y = (ap) => bottom - (ap * 100 - 46) / 18 * (bottom - top)
  const highlighted = active ? familyOf(active) : family

  return (
    <figure aria-labelledby={`${id}-title`} className="mt-6 rounded-2xl border border-surface-200 p-4 sm:p-6 dark:border-surface-800 [--chart-rfdetr:#16a34a] [--chart-yolo9:#ea580c] [--chart-yolox:#2563eb] [--chart-ec:#7c3aed] [--chart-nas:#db2777] dark:[--chart-rfdetr:#4ade80] dark:[--chart-yolo9:#fb923c] dark:[--chart-yolox:#60a5fa] dark:[--chart-ec:#a78bfa] dark:[--chart-nas:#f472b6]">
      <figcaption id={`${id}-title`} className="text-sm text-surface-600 dark:text-surface-400">
        {t('rfComplete', { count: RESULTS.length })}
      </figcaption>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2">
        {FAMILIES.map((item) => (
          <button
            key={item.name}
            type="button"
            aria-pressed={family === item.name}
            onClick={() => { setActive(null); setFamily(family === item.name ? null : item.name) }}
            className={`inline-flex min-h-8 items-center gap-2 text-xs font-medium transition-opacity focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-libre-500 ${highlighted && highlighted !== item.name ? 'opacity-40' : ''} text-surface-700 dark:text-surface-300`}
          >
            <svg width="22" height="12" aria-hidden="true" style={{ color: item.color }}>
              {item.name === 'LoRA'
                ? <path d="M11 1 L16 6 L11 11 L6 6 Z" fill="none" stroke="currentColor" strokeWidth="2" />
                : <><path d="M0 6 H22" stroke="currentColor" strokeWidth="2" /><circle cx="11" cy="6" r="3" fill="currentColor" /></>}
            </svg>
            {item.label || item.name}
          </button>
        ))}
      </div>

      <div ref={container} className="mt-2 w-full">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={height}
          role="group"
          aria-label={t('rfChartTitle')}
          className="overflow-visible text-surface-500 dark:text-surface-400"
          onMouseLeave={() => setActive(null)}
          onKeyDown={(event) => { if (event.key === 'Escape') { setActive(null); setFamily(null) } }}
        >
          <text x={left} y="17" fill="currentColor" fontSize="12">{t('rfYAxis')}</text>
          {[48, 52, 56, 60, 64].map((tick) => (
            <g key={tick}>
              <line x1={left} x2={right} y1={y(tick / 100)} y2={y(tick / 100)} className="stroke-surface-200 dark:stroke-surface-800" strokeDasharray="3 5" />
              <text x={left - 12} y={y(tick / 100) + 4} textAnchor="end" fill="currentColor" fontSize="12">{tick}</text>
            </g>
          ))}
          {[0, 10, 20, 30, 40, 50].map((tick) => (
            <text key={tick} x={x(tick)} y={bottom + 21} textAnchor="middle" fill="currentColor" fontSize="12">{tick}</text>
          ))}
          <line x1={left} x2={right} y1={bottom} y2={bottom} className="stroke-surface-200 dark:stroke-surface-800" />
          <text x={(left + right) / 2} y={height - 4} textAnchor="middle" fill="currentColor" fontSize="12">{t('rfXAxis')}</text>

          {[...SERIES].sort((a, b) => Number(a.name === family) - Number(b.name === family)).map((series) => (
            <g key={series.name} opacity={highlighted && highlighted !== series.name ? 0.15 : 1} className="transition-opacity">
              {/* Straight segments show the measured sizes, not an interpolated scaling law.
                  LoRA is a different tuning method and is deliberately not connected. */}
              {series.points.length > 1 && (
                <polyline data-family={series.name} points={series.points.map((row) => `${x(row.paramsM)},${y(row.map)}`).join(' ')} fill="none" stroke={series.color} strokeWidth="2" strokeLinejoin="round" />
              )}
              {series.points.map((row) => {
                const selected = active?.id === row.id
                const labelBelow = ['ec-m', 'ec-l', 'yolox-s'].includes(row.id)
                const labelLeft = row.id === 'rfdetr-n'
                return (
                  <g
                    key={row.id}
                    data-model={row.id}
                    role="button"
                    tabIndex={0}
                    aria-label={`${row.model}-${row.size}: ${(row.map * 100).toFixed(2)} AP50:95, ${row.paramsM}M, ${row.inputSize}px`}
                    aria-describedby={`${id}-detail`}
                    onMouseEnter={() => setActive(row)}
                    onFocus={() => setActive(row)}
                    onBlur={() => setActive(null)}
                    onClick={() => setActive(row)}
                    onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setActive(row) } }}
                    className="cursor-pointer outline-none"
                  >
                    <circle cx={x(row.paramsM)} cy={y(row.map)} r="10" fill="transparent" />
                    {selected && <circle cx={x(row.paramsM)} cy={y(row.map)} r="11" fill="none" stroke={series.color} strokeWidth="1.5" />}
                    {series.name === 'LoRA'
                      ? <path d={`M${x(row.paramsM)} ${y(row.map) - 6} l6 6 l-6 6 l-6 -6 Z`} className="fill-white dark:fill-surface-950" stroke={series.color} strokeWidth="2" />
                      : <circle cx={x(row.paramsM)} cy={y(row.map)} r="4.5" fill={series.color} className="stroke-white dark:stroke-surface-950" strokeWidth="1.5" />}
                    {/* Size labels remain legible on desktop; touch/keyboard selection
                        supplies the full model name and values at every viewport. */}
                    {!compact && (
                      <text x={x(row.paramsM) + (labelLeft ? -10 : 10)} y={y(row.map) + (labelBelow ? 18 : labelLeft || series.name === 'LoRA' ? 4 : -10)} textAnchor={labelLeft ? 'end' : 'start'} fill={series.color} fontSize="12" fontWeight="600">
                        {series.name === 'LoRA' ? 'LoRA' : row.size}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          ))}
        </svg>
      </div>
      <div id={`${id}-detail`} aria-live="polite" aria-atomic="true" className="mt-3 flex min-h-12 flex-wrap items-center gap-x-3 gap-y-1 border-t border-surface-100 pt-3 text-xs text-surface-500 dark:border-surface-800 dark:text-surface-400">
        {active ? <>
          <strong className="text-surface-900 dark:text-white">{active.model}-{active.size}</strong>
          <span className="font-mono">{(active.map * 100).toFixed(2)} AP50:95</span>
          <span className="font-mono">{active.paramsM}M</span>
          <span className="font-mono">{active.inputSize}px</span>
        </> : t('rfChartHint')}
      </div>
    </figure>
  )
}
