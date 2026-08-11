'use client'

/*
 * Copy this page as markdown, or open the markdown twin.
 *
 * "Copy page" fetches `<path>.md` rather than scraping the DOM. What lands on
 * the clipboard is then byte-identical to what an agent fetching the twin
 * receives: real code fences, generated tables expanded, no navigation chrome.
 * Scraping innerText was what made the old copy-docs button produce soup.
 *
 * These are the only interactions allowed on a docs page, under the same
 * exemption as the code-block copy button: they add convenience and hide
 * nothing. Rendered as plain text controls, because a pair of pill buttons is
 * exactly the decoration the rest of this design removed.
 */

import { useState } from 'react'
import { useTranslations } from 'next-intl'

export default function PageActions({ path }) {
  const t = useTranslations('DocsChrome')
  const [state, setState] = useState('idle')
  const markdownUrl = `${path}.md`

  const copy = async () => {
    try {
      const response = await fetch(markdownUrl)
      if (!response.ok) throw new Error(String(response.status))
      await navigator.clipboard.writeText(await response.text())
      setState('copied')
    } catch {
      setState('failed')
    }
    setTimeout(() => setState('idle'), 2000)
  }

  const label = state === 'copied' ? t('copied') : state === 'failed' ? t('copyFailed') : t('copyPage')

  return (
    <div className="flex shrink-0 items-center gap-3 text-[13px]">
      <button
        onClick={copy}
        className="text-surface-500 transition-colors hover:text-surface-900 dark:text-surface-500 dark:hover:text-surface-200"
        title={t('copyTitle')}
      >
        {label}
      </button>
      <a
        href={markdownUrl}
        className="text-surface-500 transition-colors hover:text-surface-900 dark:text-surface-500 dark:hover:text-surface-200"
        title={t('viewMarkdownTitle')}
      >
        {t('viewMarkdown')}
      </a>
    </div>
  )
}
