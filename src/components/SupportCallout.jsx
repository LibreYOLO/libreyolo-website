'use client'

import { useTranslations } from 'next-intl'
import { Star } from 'lucide-react'
import GitHubStarButton from './GitHubStarButton'

const REPO_URL = 'https://github.com/LibreYOLO/libreyolo'

export default function SupportCallout({ className = '' }) {
  const t = useTranslations('Support')

  const link = (href) =>
    function LinkChunk(chunks) {
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-libre-600 dark:text-libre-400 hover:underline"
        >
          {chunks}
        </a>
      )
    }

  return (
    <div
      className={`rounded-xl border border-surface-200 dark:border-white/10 bg-white dark:bg-white/[0.03] p-6 sm:p-8 flex flex-col sm:flex-row items-start sm:items-center gap-6 ${className}`}
    >
      <div className="flex-1">
        <h2 className="flex items-center gap-2 text-base font-semibold text-surface-900 dark:text-white mb-2">
          <Star className="w-4 h-4 text-libre-500 dark:text-libre-400" />
          {t('title')}
        </h2>
        <p className="text-sm leading-relaxed text-surface-600 dark:text-surface-300">
          {t.rich('body', {
            star: link(REPO_URL),
            issue: link(`${REPO_URL}/issues`),
            contrib: link(`${REPO_URL}/blob/HEAD/CONTRIBUTING.md`),
          })}
        </p>
      </div>
      <GitHubStarButton className="shrink-0" />
    </div>
  )
}
