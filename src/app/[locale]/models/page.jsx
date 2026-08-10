'use client'

import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { X, Maximize2 } from 'lucide-react'
import { Link } from '@/i18n/navigation'
import { getModelIndex, getModelStats, tierMeta } from '@/lib/models-index'

const INDEX = getModelIndex()
const STATS = getModelStats()

// Tier order for the filter, coarsest split first. Museum sits last because
// almost nobody arrives here looking for VGG.
const TIER_ORDER = ['g0', 'g1', 'g2', 's', 'g3', 'g4']

// Small corner hint so people (especially on touch) know a tile opens larger.
function ExpandBadge() {
  return (
    <span className="pointer-events-none absolute top-2 right-2 w-8 h-8 rounded-lg bg-black/40 text-white flex items-center justify-center opacity-70 group-hover:opacity-100 transition-opacity backdrop-blur-sm">
      <Maximize2 className="w-4 h-4" />
    </span>
  )
}

// Connection-friendly video: poster shows instantly, the clip is only fetched
// when scrolled into view (skipped entirely on Data Saver / 2G), pauses off-screen.
// Clicking opens the lightbox rather than navigating away.
function VideoTile({ src, poster, label, models, onOpen }) {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.muted = true

    const c = typeof navigator !== 'undefined' ? navigator.connection : null
    const saveData = !!(c && (c.saveData || /(^|\b)(slow-2g|2g)$/.test(c.effectiveType || '')))
    if (saveData) return // stay on the poster; the lightbox loads it on demand

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (!el.getAttribute('src')) el.setAttribute('src', src)
          el.play().catch(() => {})
        } else {
          el.pause()
        }
      },
      { rootMargin: '250px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [src])

  return (
    <div>
      <p className="mb-2 text-center text-sm font-medium text-surface-700 dark:text-surface-200">{label}</p>
      <button
        type="button"
        onClick={() => onOpen({ kind: 'video', src, poster, alt: label })}
        aria-label={`Enlarge ${label} demo`}
        className="group relative block w-full overflow-hidden rounded-xl border border-surface-200 dark:border-white/10 bg-surface-100 dark:bg-surface-900/60 cursor-zoom-in"
      >
        <video ref={ref} className="w-full aspect-video object-cover block" poster={poster} loop muted playsInline preload="none" />
        <ExpandBadge />
      </button>
      <p className="mt-2 text-center text-xs text-surface-500 dark:text-surface-400 leading-relaxed">
        {models}
      </p>
    </div>
  )
}

// One model. The name carries the docs link because every family has a docs
// page; weights are a separate link and are simply absent for the 19 families
// with nothing published on Hugging Face.
function ModelRow({ model, tierLabel, weightsLabel }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-surface-100 py-2.5 last:border-b-0 dark:border-white/5">
      <div className="min-w-0">
        <Link
          href={model.docsUrl}
          className="text-sm font-medium text-surface-800 underline-offset-2 hover:text-libre-600 hover:underline dark:text-surface-100 dark:hover:text-libre-400"
        >
          {model.name}
        </Link>
        {model.sizesLabel && (
          <p className="mt-0.5 truncate text-xs text-surface-400 dark:text-surface-500" title={model.sizesLabel}>
            {model.sizesLabel}
          </p>
        )}
      </div>
      <div className="flex shrink-0 items-baseline gap-3">
        <span className="font-mono text-[10px] uppercase tracking-wider text-surface-400 dark:text-surface-500">
          {tierLabel}
        </span>
        {model.hfUrl && (
          <a
            href={model.hfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-surface-500 underline-offset-2 hover:text-libre-600 hover:underline dark:text-surface-400 dark:hover:text-libre-400"
          >
            {weightsLabel}
          </a>
        )}
      </div>
    </div>
  )
}

// Artwork where a task has it, a plain framed caption where it does not. Nine
// tasks have no image yet; an empty frame reads as deliberate, a stretched
// stand-in image would not.
function TaskArt({ group, onOpen }) {
  const frame =
    'relative block w-full overflow-hidden rounded-xl border border-surface-200 bg-surface-100 dark:border-white/10 dark:bg-surface-900/60'

  if (group.video) {
    return (
      <button
        type="button"
        onClick={() => onOpen({ kind: 'video', src: group.video, poster: group.poster, alt: group.label })}
        aria-label={`Enlarge ${group.label} demo`}
        className={`group cursor-zoom-in ${frame}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={group.poster} alt={group.label} loading="lazy" className="block aspect-video w-full object-cover" />
        <ExpandBadge />
      </button>
    )
  }

  if (group.image) {
    return (
      <button
        type="button"
        onClick={() => onOpen({ kind: 'image', src: group.image, alt: group.label })}
        aria-label={`Enlarge ${group.label} example`}
        className={`group cursor-zoom-in ${frame}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={group.image}
          alt={group.label}
          loading="lazy"
          className="block aspect-video w-full object-cover transition-transform duration-500 group-hover:scale-[1.02]"
        />
        <ExpandBadge />
      </button>
    )
  }

  return (
    <div className={`${frame} flex aspect-video items-center justify-center`}>
      <span className="px-4 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-surface-400 dark:text-surface-600">
        {group.label}
      </span>
    </div>
  )
}

// In-page zoom overlay. Closes on backdrop click, the X button, or Escape.
function Lightbox({ item, onClose }) {
  useEffect(() => {
    if (!item) return
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [item, onClose])

  return (
    <AnimatePresence>
      {item && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 sm:p-8"
          onClick={onClose}
          role="dialog"
          aria-modal="true"
        >
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute top-4 right-4 w-11 h-11 rounded-full bg-white/10 hover:bg-white/25 text-white flex items-center justify-center backdrop-blur-sm transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
          <motion.div
            initial={{ scale: 0.94, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ type: 'spring', damping: 26, stiffness: 300 }}
            className="relative"
            onClick={(e) => e.stopPropagation()}
          >
            {item.kind === 'video' ? (
              <video
                src={item.src}
                poster={item.poster}
                autoPlay
                loop
                muted
                playsInline
                controls
                className="max-w-[92vw] max-h-[86vh] rounded-xl block shadow-2xl"
              />
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={item.src}
                alt={item.alt}
                className="max-w-[92vw] max-h-[86vh] rounded-xl block object-contain shadow-2xl"
              />
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default function Models() {
  const t = useTranslations('Models')
  const [lightbox, setLightbox] = useState(null)
  const [tier, setTier] = useState('all')
  const closeLightbox = useCallback(() => setLightbox(null), [])

  // Task label and blurb come from the message catalogue so the page reads in
  // both locales; the registry-derived English in models-index is the fallback
  // for any task added to the library before its translation lands.
  const label = useCallback(
    (group) => {
      const key = `tasks.${group.task}.label`
      const translated = t.has?.(key) ? t(key) : null
      return translated || group.label
    },
    [t],
  )
  const blurb = useCallback(
    (group) => {
      const key = `tasks.${group.task}.blurb`
      const translated = t.has?.(key) ? t(key) : null
      return translated || group.blurb || ''
    },
    [t],
  )

  const videoGroups = useMemo(() => INDEX.filter((g) => g.video), [])

  const filtered = useMemo(() => {
    if (tier === 'all') return INDEX
    return INDEX.map((g) => ({ ...g, models: g.models.filter((m) => m.tier === tier) })).filter(
      (g) => g.models.length > 0,
    )
  }, [tier])

  const shown = filtered.reduce((n, g) => n + g.models.length, 0)

  return (
    <div className="pt-28 lg:pt-36 pb-20">
      <div className="max-w-6xl mx-auto px-5 md:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10 md:mb-14"
        >
          <h1 className="text-2xl sm:text-3xl font-semibold text-surface-800 dark:text-white mb-2">
            {t('title')}
          </h1>
          <p className="text-sm text-surface-500 dark:text-surface-400">{t('subtitle')}</p>
        </motion.div>

        {/* Three tasks in motion, each with its model count underneath */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 md:gap-6">
          {videoGroups.map((g) => (
            <VideoTile
              key={g.video}
              src={g.video}
              poster={g.poster}
              label={label(g)}
              models={t('familyCount', { count: g.models.length })}
              onOpen={setLightbox}
            />
          ))}
        </div>

        {/* Everything the library ships, straight from the docs registry */}
        <div className="mt-20">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-surface-800 dark:text-white sm:text-2xl">
              {t('byTask')}
            </h2>
            <p className="mt-2 text-sm text-surface-500 dark:text-surface-400">
              {t('indexLead', {
                families: STATS.families,
                tasks: STATS.tasks,
                version: STATS.version,
              })}
            </p>
          </div>

          {/* Tier filter */}
          <div className="mt-6 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
            <button
              type="button"
              onClick={() => setTier('all')}
              className={`font-mono text-[11px] uppercase tracking-wider transition-colors ${
                tier === 'all'
                  ? 'text-libre-600 dark:text-libre-400'
                  : 'text-surface-400 hover:text-surface-700 dark:text-surface-500 dark:hover:text-surface-300'
              }`}
            >
              {t('allTiers')}
            </button>
            {TIER_ORDER.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => setTier(id)}
                title={tierMeta(id).blurb}
                className={`font-mono text-[11px] uppercase tracking-wider transition-colors ${
                  tier === id
                    ? 'text-libre-600 dark:text-libre-400'
                    : 'text-surface-400 hover:text-surface-700 dark:text-surface-500 dark:hover:text-surface-300'
                }`}
              >
                {tierMeta(id).label}
              </button>
            ))}
          </div>

          {tier !== 'all' && (
            <p className="mt-3 text-center text-xs text-surface-400 dark:text-surface-500">
              {tierMeta(tier).blurb}
            </p>
          )}

          <p className="mt-3 text-center font-mono text-[11px] text-surface-400 dark:text-surface-600">
            {t('showing', { shown })}
          </p>

          <div className="mt-10 space-y-14">
            {filtered.map((group) => (
              <section key={group.task}>
                <div className="grid gap-6 md:grid-cols-[minmax(0,280px)_1fr] md:gap-8">
                  <div>
                    <TaskArt group={{ ...group, label: label(group) }} onOpen={setLightbox} />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-baseline justify-between gap-4">
                      <h3 className="text-lg font-semibold text-surface-800 dark:text-white">
                        {label(group)}
                      </h3>
                      <span className="shrink-0 font-mono text-[11px] text-surface-400 dark:text-surface-600">
                        {t('familyCount', { count: group.models.length })}
                      </span>
                    </div>
                    {blurb(group) && (
                      <p className="mt-1 text-sm text-surface-500 dark:text-surface-400">{blurb(group)}</p>
                    )}
                    <div className="mt-4 grid gap-x-8 sm:grid-cols-2">
                      {group.models.map((m) => (
                        <ModelRow
                          key={`${group.task}-${m.key}`}
                          model={m}
                          tierLabel={tierMeta(m.tier).label}
                          weightsLabel={t('weightsShort')}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </section>
            ))}
          </div>
        </div>

        <div className="mt-16 flex items-center justify-center gap-8 text-sm">
          <a
            href="https://huggingface.co/LibreYOLO/models"
            target="_blank"
            rel="noopener noreferrer"
            className="text-surface-500 hover:text-libre-600 dark:hover:text-libre-400 transition-colors"
          >
            {t('weights')} &#8599;
          </a>
          <Link
            href="/docs"
            className="text-surface-500 hover:text-libre-600 dark:hover:text-libre-400 transition-colors"
          >
            {t('docs')} &#8594;
          </Link>
        </div>
      </div>

      <Lightbox item={lightbox} onClose={closeLightbox} />
    </div>
  )
}
