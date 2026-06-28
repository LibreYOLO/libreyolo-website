'use client'

import { useTransition } from 'react'
import { useLocale, useTranslations } from 'next-intl'
import { usePathname, useRouter } from '@/i18n/navigation'

// Compact EN / 中文 toggle. Switches locale while staying on the current path,
// so a reader on /zh/models lands on /models when they pick English (and back).
export default function LanguageSwitcher({ className = '' }) {
  const locale = useLocale()
  const pathname = usePathname()
  const router = useRouter()
  const t = useTranslations('LangSwitcher')
  const [isPending, startTransition] = useTransition()

  const switchTo = (next) => {
    if (next === locale) return
    startTransition(() => {
      router.replace(pathname, { locale: next })
    })
  }

  const options = [
    { code: 'en', label: 'EN' },
    { code: 'zh', label: '中文' },
  ]

  return (
    <div
      role="group"
      aria-label={t('label')}
      className={`inline-flex items-center rounded-lg border border-surface-200 dark:border-white/10 p-0.5 ${className}`}
    >
      {options.map((opt) => {
        const isActive = opt.code === locale
        return (
          <button
            key={opt.code}
            type="button"
            onClick={() => switchTo(opt.code)}
            disabled={isPending}
            aria-pressed={isActive}
            lang={opt.code === 'zh' ? 'zh-CN' : 'en'}
            className={`px-2 py-1 rounded-md text-xs font-medium transition-colors ${
              isActive
                ? 'bg-libre-500/10 text-libre-600 dark:text-libre-400'
                : 'text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white'
            }`}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
