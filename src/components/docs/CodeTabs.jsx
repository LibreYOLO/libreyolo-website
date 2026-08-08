/*
 * A group of related snippets, rendered stacked and all visible.
 *
 * This used to be a tab strip. Tabs hide every variant except one behind a
 * click, and the two readers that matter most here cannot click: an agent
 * reading the page or its .md twin, and a person scanning for the CLI form who
 * has no way to know it exists. Nothing on a docs page should require an
 * interaction to read. Stacking costs vertical space and returns everything
 * else: the page is complete in one pass, in the DOM, in the markdown twin,
 * and in a print or screen-reader rendering.
 *
 * Server component; no state, no client bundle beyond each block's copy button.
 */

import Code from '@/components/docs/Code'

export default function CodeTabs({ tabs = [] }) {
  if (!tabs.length) return null

  return (
    <div className="my-4 flex flex-col gap-3">
      {tabs.map((tab) => (
        <div key={tab.label}>
          <Code language={tab.language} label={tab.label}>
            {tab.code}
          </Code>
          {tab.expect && (
            <div className="-mt-1">
              <p className="mb-1 text-[12px] text-surface-500 dark:text-surface-500">Output</p>
              <pre className="overflow-x-auto border-l-2 border-surface-300 py-0.5 pl-3 font-mono text-[12px] leading-[1.7] text-surface-600 dark:border-white/20 dark:text-surface-400">
                {tab.expect.replace(/\n$/, '')}
              </pre>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
