'use client'

/*
 * Python / CLI tab pair.
 *
 * Every runnable snippet on a docs page comes from page frontmatter rather than
 * from the markdown body, so the same strings can be extracted and executed by
 * snippet CI. This component only renders them.
 */

import { useState } from 'react'
import Code from '@/components/docs/Code'

export default function CodeTabs({ tabs = [] }) {
  const [active, setActive] = useState(0)
  if (!tabs.length) return null
  const current = tabs[Math.min(active, tabs.length - 1)]

  return (
    <div className="my-4">
      {/* Tabs are underline-switched text, not a segmented control. */}
      <div className="flex items-center gap-4 border-b border-surface-200 dark:border-white/[0.09]" role="tablist">
        {tabs.map((tab, index) => (
          <button
            key={tab.label}
            role="tab"
            aria-selected={index === active}
            onClick={() => setActive(index)}
            className={`-mb-px border-b-2 pb-1.5 text-[13px] transition-colors ${
              index === active
                ? 'border-surface-800 font-medium text-surface-900 dark:border-surface-300 dark:text-white'
                : 'border-transparent text-surface-500 hover:text-surface-800 dark:text-surface-500 dark:hover:text-surface-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <Code language={current.language}>{current.code}</Code>
      {current.expect && (
        <div className="-mt-2 mb-4">
          <p className="mb-1 text-[12px] text-surface-500 dark:text-surface-500">Output</p>
          <pre className="overflow-x-auto border-l-2 border-surface-300 py-0.5 pl-3 font-mono text-[12px] leading-[1.7] text-surface-600 dark:border-white/20 dark:text-surface-400">
            {current.expect.replace(/\n$/, '')}
          </pre>
        </div>
      )}
    </div>
  )
}
