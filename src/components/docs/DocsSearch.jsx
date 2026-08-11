'use client'

/*
 * Docs search: a build-time index, matched in the browser.
 *
 * The index (/docs/search-index.json) is fetched once, on first open, not on
 * page load, so a reader who never searches pays nothing for it.
 *
 * Scoring is hand-rolled rather than a fuzzy-match dependency. On a corpus of
 * this size the useful behaviour is narrow and specific: an exact title match
 * must win, a prefix must beat a substring, and a heading hit must be able to
 * deep-link to its anchor. A generic fuzzy library ranks "obb" above
 * "Oriented boxes" on edit distance and cannot do the anchor part at all.
 *
 * Every term must match somewhere (AND, not OR), because "rf-detr export"
 * should mean both words, which is what people expect from a docs box.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useTranslations } from 'next-intl'
import { Search, CornerDownLeft } from 'lucide-react'
import { slugifyHeading } from '@/lib/slugify-heading'

/*
 * How well `term` matches inside `text`, 0 to 1.
 *
 * Plain substring matching is wrong here in a way that is easy to miss:
 * "torch" is a substring of "TorchScript", so a search for "torch free"
 * ranked an unrelated detector above the torch-free install page. Whole-word
 * hits therefore have to outrank word-prefix hits, which in turn outrank a
 * substring landing in the middle of a word. Prefixes still count, because a
 * reader typing "orient" expects to see "Oriented boxes" before they finish.
 */
function matchStrength(text, term) {
  const t = text.toLowerCase()
  if (!t.includes(term)) return 0
  const word = new RegExp(`(?:^|[^a-z0-9])${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:[^a-z0-9]|$)`)
  if (word.test(t)) return 1
  const prefix = new RegExp(`(?:^|[^a-z0-9])${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`)
  if (prefix.test(t)) return 0.6
  return 0.25
}

// Higher is better. Ties break on the page title so results are stable.
function scoreEntry(entry, terms) {
  const title = entry.t.toLowerCase()
  const desc = entry.d || ''
  const headings = entry.h || []
  let total = 0
  let anchor = null

  for (const term of terms) {
    let best = title === term ? 100 : 62 * matchStrength(title, term)

    for (const h of headings) {
      const hit = h.toLowerCase() === term ? 40 : 34 * matchStrength(h, term)
      if (hit > best) { best = hit; anchor = h }
      // A heading match only supplies the anchor when nothing on the title
      // scored higher, otherwise "export" on the RF-DETR page would jump past
      // the introduction for a reader who searched the family name.
    }

    // Keywords score below headings and above description: they are declared
    // synonyms for the page, not text a reader will see on it, so they should
    // pull a page into the results without outranking a real heading hit.
    for (const k of entry.k || []) best = Math.max(best, 22 * matchStrength(k, term))
    best = Math.max(best, 12 * matchStrength(desc, term))
    if (!best) return null // every term must land somewhere
    total += best
  }

  return { entry, score: total, anchor }
}

export default function DocsSearch() {
  const t = useTranslations('DocsChrome')
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(null)
  const [active, setActive] = useState(0)
  const inputRef = useRef(null)
  const sectionLabels = Object.fromEntries(
    ['start', 'tasks', 'models', 'train', 'predict', 'export', 'cli', 'reference']
      .map((id) => [id, t(`groups.${id}`)])
  )

  const load = useCallback(async () => {
    if (index) return
    try {
      const res = await fetch('/docs/search-index.json')
      setIndex(res.ok ? await res.json() : [])
    } catch {
      setIndex([])
    }
  }, [index])

  // Cmd/Ctrl+K opens, / opens when not already typing, Escape closes.
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((v) => !v)
        return
      }
      if (e.key === 'Escape') return setOpen(false)
      const typing = /^(INPUT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable
      if (e.key === '/' && !typing) {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (!open) return
    load()
    // The input mounts with the dialog, so focus after paint.
    const id = requestAnimationFrame(() => inputRef.current?.focus())
    return () => cancelAnimationFrame(id)
  }, [open, load])

  const results = useMemo(() => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
    if (!index || !terms.length) return []
    return index
      .map((e) => scoreEntry(e, terms))
      .filter(Boolean)
      .sort((a, b) => b.score - a.score || a.entry.t.localeCompare(b.entry.t))
      .slice(0, 12)
  }, [index, query])

  useEffect(() => { setActive(0) }, [query])

  const go = (hit) => {
    if (!hit) return
    setOpen(false)
    setQuery('')
    router.push(hit.anchor ? `${hit.entry.p}#${slugifyHeading(hit.anchor)}` : hit.entry.p)
  }

  const onInputKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((i) => Math.min(i + 1, results.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); go(results[active]) }
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex w-full items-center gap-2 rounded-lg border border-surface-200 px-3 py-2 text-[13px] text-surface-500 transition-colors hover:border-surface-300 dark:border-white/[0.08] dark:text-surface-500 dark:hover:border-white/20"
      >
        <Search className="h-3.5 w-3.5 shrink-0" />
        <span>{t('searchTrigger')}</span>
        <kbd className="ml-auto hidden font-sans text-[11px] text-surface-400 dark:text-surface-600 sm:block">
          /
        </kbd>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-[60] flex items-start justify-center bg-black/50 p-4 pt-[12vh] backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label={t('searchDialogLabel')}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-xl overflow-hidden rounded-xl border border-surface-200 bg-white shadow-2xl dark:border-white/[0.08] dark:bg-surface-950"
          >
            <div className="flex items-center gap-2.5 border-b border-surface-200 px-4 dark:border-white/[0.06]">
              <Search className="h-4 w-4 shrink-0 text-surface-400 dark:text-surface-600" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onInputKey}
                placeholder={t('searchPlaceholder')}
                aria-label={t('searchPlaceholder')}
                className="w-full bg-transparent py-3.5 text-[15px] text-surface-900 outline-none placeholder:text-surface-400 dark:text-white dark:placeholder:text-surface-600"
              />
            </div>

            {query && (
              <ul className="max-h-[55vh] overflow-y-auto py-1.5">
                {results.length === 0 && (
                  <li className="px-4 py-6 text-center text-[13.5px] text-surface-500 dark:text-surface-500">
                    {index === null ? t('searchLoading') : t('searchNoResults', { query })}
                  </li>
                )}
                {results.map((hit, i) => (
                  <li key={hit.entry.p + (hit.anchor || '')}>
                    <button
                      onMouseEnter={() => setActive(i)}
                      onClick={() => go(hit)}
                      className={`flex w-full items-center gap-3 px-4 py-2.5 text-left ${
                        i === active ? 'bg-surface-100 dark:bg-white/[0.06]' : ''
                      }`}
                    >
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[14px] font-medium text-surface-900 dark:text-white">
                          {hit.entry.t}
                          {hit.anchor && (
                            <span className="font-normal text-surface-500 dark:text-surface-500">
                              {' '}› {hit.anchor}
                            </span>
                          )}
                        </span>
                        <span className="block truncate text-[12.5px] text-surface-500 dark:text-surface-500">
                          {sectionLabels[hit.entry.s] || hit.entry.s}
                        </span>
                      </span>
                      {i === active && (
                        <CornerDownLeft className="h-3.5 w-3.5 shrink-0 text-surface-400 dark:text-surface-600" />
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </>
  )
}
