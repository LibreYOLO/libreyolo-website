'use client'

import { useTranslations } from 'next-intl'
import { Star } from 'lucide-react'
import { GithubIcon, RedditIcon } from './BrandIcons'
import { GITHUB_URL as REPO_URL, REDDIT_URL } from '@/lib/links'

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
        <p className="text-sm leading-relaxed text-surface-600 dark:text-surface-300 mt-2">
          {t('communityBody')}
        </p>
      </div>
      <div className="shrink-0 flex flex-col gap-2 w-full sm:w-auto">
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-surface-200 dark:border-white/10 px-3.5 py-2 text-sm font-semibold text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 hover:bg-surface-50 dark:hover:bg-white/5 transition-colors whitespace-nowrap"
        >
          <GithubIcon className="w-4 h-4" />
          {t('githubCta')}
        </a>
        <a
          href={REDDIT_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-[#FF4500]/30 bg-[#FF4500]/10 px-3.5 py-2 text-sm font-semibold text-[#D93A00] dark:text-[#FF6A33] hover:bg-[#FF4500]/20 transition-colors whitespace-nowrap"
        >
          <RedditIcon className="w-4 h-4" />
          {t('communityCta')}
        </a>
      </div>
    </div>
  )
}
