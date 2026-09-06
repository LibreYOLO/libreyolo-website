'use client'

import { useEffect, useRef, useState } from 'react'
import { useTranslations } from 'next-intl'

export default function ModelArchitecture({ diagram }) {
  const t = useTranslations('ModelDiagram')
  const frame = useRef(null)
  const [selected, setSelected] = useState(diagram.default_view)
  const [height, setHeight] = useState(1600)
  const view = diagram.views.find(v => v.id === selected) || diagram.views[0]
  const title = t('title', { model: diagram.title })

  useEffect(() => {
    function onSize(event) {
      if (event.origin !== window.location.origin || event.source !== frame.current?.contentWindow) return
      if (event.data?.type === 'libreyolo-architecture-height' && Number.isFinite(event.data.height)) {
        setHeight(Math.max(400, Math.min(20000, event.data.height)))
      }
    }
    window.addEventListener('message', onSize)
    return () => window.removeEventListener('message', onSize)
  }, [])

  function sizeOnLoad() {
    const body = frame.current?.contentDocument?.body
    if (body) setHeight(Math.max(400, Math.min(20000, Math.ceil(body.getBoundingClientRect().height) + 2)))
  }

  return (
    <section id="architecture" aria-labelledby="architecture-title" className="mt-14 scroll-mt-24 border-t border-surface-200 pt-6 dark:border-white/[0.09]">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="architecture-title" className="text-2xl font-semibold text-surface-900 dark:text-white">{title}</h2>
        <div className="flex gap-4 text-sm text-libre-700 underline dark:text-libre-400">
          <a href={view.html} target="_blank" rel="noopener noreferrer">{t('open')}</a>
          <a href={view.svg} download>SVG</a>
        </div>
      </div>
      <p className="mb-4 max-w-[70ch] text-sm text-surface-600 dark:text-surface-400">{t('description')}</p>
      {diagram.views.length > 1 && (
        <label className="mb-4 flex flex-wrap items-center gap-3 text-sm text-surface-700 dark:text-surface-300">
          {t('view')}
          <select value={view.id} onChange={event => {setSelected(event.target.value); setHeight(1600)}}
            className="max-w-full rounded border border-surface-300 bg-white px-3 py-2 text-surface-900 dark:border-white/20 dark:bg-surface-900 dark:text-white">
            {diagram.views.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
      )}
      <iframe key={view.id} ref={frame} src={view.html} title={`${title}: ${view.label}`} loading="lazy" onLoad={sizeOnLoad}
        className="block w-full border border-surface-200 bg-white dark:border-white/[0.09]" style={{ height }} />
    </section>
  )
}
