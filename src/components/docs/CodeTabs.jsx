'use client'

/*
 * Python / CLI tab pair.
 *
 * Every runnable snippet on a docs page comes from page frontmatter rather than
 * from the markdown body, so the same strings can be extracted and executed by
 * snippet CI. This component only renders them.
 */

import { useState } from 'react'
import { CodeBlock } from '@/components/DocsKit'

export default function CodeTabs({ tabs = [] }) {
  const [active, setActive] = useState(0)
  if (!tabs.length) return null
  const current = tabs[Math.min(active, tabs.length - 1)]

  return (
    <div className="my-5">
      <div className="flex items-center gap-1 rounded-lg border border-surface-200 bg-surface-50 p-1 dark:border-white/[0.08] dark:bg-white/[0.02]" role="tablist">
        {tabs.map((tab, index) => (
          <button
            key={tab.label}
            role="tab"
            aria-selected={index === active}
            onClick={() => setActive(index)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              index === active
                ? 'bg-white text-surface-900 shadow-sm dark:bg-white/[0.08] dark:text-white'
                : 'text-surface-500 hover:text-surface-900 dark:text-surface-400 dark:hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <CodeBlock language={current.language}>{current.code.replace(/\n$/, '')}</CodeBlock>
      {current.expect && (
        <div className="-mt-2 rounded-b-xl border border-t-0 border-surface-200 bg-surface-50 px-4 py-3 dark:border-white/[0.08] dark:bg-white/[0.02]">
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-surface-500 dark:text-surface-500">Expected output</p>
          <pre className="overflow-x-auto font-mono text-xs leading-relaxed text-surface-600 dark:text-surface-400">{current.expect.replace(/\n$/, '')}</pre>
        </div>
      )}
    </div>
  )
}
