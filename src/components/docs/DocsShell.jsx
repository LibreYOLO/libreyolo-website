'use client'

/*
 * Layout shell for the exploded docs (v2).
 *
 * Three columns on xl: nav rail, content, "on this page". Two on lg (no right
 * rail), one on mobile with the nav behind a drawer. The old versioned docs
 * pages keep their own single-file shell; this one is for content-collection
 * pages under /docs/<section>/<slug>.
 */

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, ChevronRight, ChevronDown, Hash } from 'lucide-react'
import PageActions from '@/components/docs/PageActions'
import DocsSearch from '@/components/docs/DocsSearch'
import { DOCS_PRERELEASE, DOCS_CURRENT_VERSION } from '@/data/docs-versions'

function NavGroup({ group, activePath, onNavigate }) {
  const containsActive = group.items.some((item) => item.slug === activePath)
  const [open, setOpen] = useState(containsActive)

  return (
    <li>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-3 py-2 rounded-lg text-[13px] font-semibold uppercase tracking-wide text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white transition-colors"
        aria-expanded={open}
      >
        <ChevronRight
          className={`w-3.5 h-3.5 shrink-0 transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
        />
        {group.label}
      </button>

      {open && (
        <ul className="ml-[19px] border-l border-surface-200 dark:border-white/[0.08] pl-2 py-0.5 space-y-px">
          {group.items.map((item) => {
            const isActive = item.slug === activePath
            if (!item.built) {
              return (
                <li key={item.slug}>
                  <span
                    className="block py-1 pl-3 text-[13.5px] text-surface-400/80 dark:text-surface-600 cursor-default select-none"
                    title="Planned, not written yet"
                  >
                    {item.label}
                  </span>
                </li>
              )
            }
            return (
              <li key={item.slug}>
                <Link
                  href={item.slug}
                  onClick={onNavigate}
                  aria-current={isActive ? 'page' : undefined}
                  className={`-ml-px block border-l-2 py-1 pl-3 text-[13.5px] transition-colors ${
                    isActive
                      ? 'border-surface-800 font-medium text-surface-900 dark:border-surface-300 dark:text-white'
                      : 'border-transparent text-surface-600 hover:text-surface-900 dark:text-surface-400 dark:hover:text-white'
                  }`}
                >
                  {item.label}
                </Link>
              </li>
            )
          })}
          {group.more && (
            <li className="px-2.5 py-1.5 text-xs italic text-surface-400 dark:text-surface-600">
              {group.more}
            </li>
          )}
        </ul>
      )}
    </li>
  )
}

function NavTree({ nav, activePath, version, onNavigate }) {
  return (
    <div>
      <div className="px-3 pb-4 mb-2 border-b border-surface-200 dark:border-white/[0.06]">
        <Link href="/docs" onClick={onNavigate} className="block group">
          <span className="text-sm font-bold text-surface-900 dark:text-white">Documentation</span>
        </Link>
        {/*
          The rail states which release these pages describe, and it must not
          claim a release exists before it is tagged. This tree documents the
          development branch, which carries model families and export formats
          the current release does not have, so labelling it with the current
          version number would tell a reader that `pip install libreyolo`
          ships things it does not. Until the version is cut, the rail says so
          and links the frozen page for the release people are actually
          running. Flip `DOCS_PRERELEASE` to false on tag day.
        */}
        {DOCS_PRERELEASE ? (
          <span className="mt-1 flex flex-col gap-0.5 text-xs font-medium">
            <span className="inline-flex items-center gap-1.5 text-surface-500 dark:text-surface-500">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              Next release
            </span>
            <Link
              href={`/docs/v${DOCS_CURRENT_VERSION}`}
              onClick={onNavigate}
              className="pl-3 text-surface-400 underline-offset-2 hover:underline dark:text-surface-600"
            >
              v{DOCS_CURRENT_VERSION} is current
            </Link>
          </span>
        ) : (
          <span className="mt-1 inline-flex items-center gap-1.5 text-xs font-medium text-surface-500 dark:text-surface-500">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            v{version}
            <span className="text-surface-400 dark:text-surface-600">latest</span>
          </span>
        )}
        <div className="mt-3">
          <DocsSearch />
        </div>
      </div>
      <ul className="space-y-px">
        {nav.groups.map((group) => (
          <NavGroup key={group.id} group={group} activePath={activePath} onNavigate={onNavigate} />
        ))}
      </ul>
    </div>
  )
}

function OnThisPage({ headings }) {
  const [active, setActive] = useState(headings[0]?.id)

  useEffect(() => {
    const onScroll = () => {
      const threshold = 140
      let current = headings[0]?.id
      for (const { id } of headings) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= threshold) current = id
      }
      setActive(current)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()
    return () => window.removeEventListener('scroll', onScroll)
  }, [headings])

  return (
    <nav aria-label="On this page">
      <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-surface-500 dark:text-surface-500 mb-3">
        <Hash className="w-3 h-3" />
        On this page
      </p>
      <ul className="space-y-px border-l border-surface-200 dark:border-white/[0.08]">
        {headings.map(({ id, title }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={`block -ml-px border-l-2 pl-3 py-1 text-[13px] leading-snug transition-colors ${
                active === id
                  ? 'border-libre-500 text-libre-700 dark:text-libre-300 font-medium'
                  : 'border-transparent text-surface-500 dark:text-surface-500 hover:text-surface-900 dark:hover:text-surface-200'
              }`}
            >
              {title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}

export default function DocsShell({ nav, activePath, version, headings = [], breadcrumbs = [], showActions = true, children }) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const closeDrawer = () => setDrawerOpen(false)

  return (
    <div className="mx-auto max-w-[1600px] px-0 lg:px-6">
      <div className="flex gap-0 lg:gap-8">
        {/* Nav rail */}
        <aside className="hidden lg:block shrink-0 w-64 xl:w-72">
          <div className="sticky top-24 max-h-[calc(100vh-7rem)] overflow-y-auto py-10 pr-2">
            <NavTree nav={nav} activePath={activePath} version={version} />
          </div>
        </aside>

        {/* Content */}
        <div className="min-w-0 flex-1 px-6 lg:px-0 pt-28 pb-24">
          {/* Breadcrumb on the left, page actions on the right, sharing one row
              so neither needs a band of its own. */}
          <div className="mb-6 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
            {breadcrumbs.length > 0 ? (
            <nav aria-label="Breadcrumb">
              <ol className="flex flex-wrap items-center gap-1.5 text-[13px] text-surface-500 dark:text-surface-500">
                {breadcrumbs.map((crumb, index) => (
                  <li key={crumb.href || crumb.label} className="flex items-center gap-1.5">
                    {index > 0 && <ChevronRight className="w-3.5 h-3.5 text-surface-300 dark:text-surface-700" />}
                    {crumb.href ? (
                      <Link href={crumb.href} className="hover:text-libre-600 dark:hover:text-libre-400 transition-colors">
                        {crumb.label}
                      </Link>
                    ) : (
                      <span className="text-surface-700 dark:text-surface-300 font-medium">{crumb.label}</span>
                    )}
                  </li>
                ))}
              </ol>
            </nav>
            ) : <span />}
            {showActions && <PageActions path={activePath} />}
          </div>
          {children}
        </div>

        {/* On this page */}
        {headings.length > 0 && (
          <aside className="hidden xl:block shrink-0 w-56">
            <div className="sticky top-24 max-h-[calc(100vh-7rem)] overflow-y-auto py-10">
              <OnThisPage headings={headings} />
            </div>
          </aside>
        )}
      </div>

      {/* Mobile nav */}
      <button
        onClick={() => setDrawerOpen(true)}
        className="lg:hidden fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-surface-900 dark:bg-white px-4 py-3 text-sm font-semibold text-white dark:text-surface-900 shadow-lg shadow-surface-900/20"
        aria-label="Open documentation navigation"
      >
        <Menu className="w-4 h-4" />
        Docs
      </button>

      <AnimatePresence>
        {drawerOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={closeDrawer}
              className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm lg:hidden"
            />
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: 'spring', damping: 28, stiffness: 280 }}
              className="fixed inset-y-0 left-0 z-50 w-[19rem] overflow-y-auto border-r border-surface-200 bg-white px-4 py-6 dark:border-white/[0.08] dark:bg-surface-950 lg:hidden"
            >
              <div className="mb-4 flex justify-end">
                <button
                  onClick={closeDrawer}
                  className="rounded-lg p-2 text-surface-500 hover:bg-surface-100 hover:text-surface-900 dark:hover:bg-white/[0.06] dark:hover:text-white"
                  aria-label="Close navigation"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <NavTree nav={nav} activePath={activePath} version={version} onNavigate={closeDrawer} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}

export { OnThisPage, ChevronDown }
