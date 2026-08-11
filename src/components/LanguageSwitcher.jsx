'use client'

import { useEffect, useRef, useState, useTransition } from 'react'
import { useLocale, useTranslations } from 'next-intl'
import { Check, Globe } from 'lucide-react'
import { usePathname, useRouter } from '@/i18n/navigation'
import { routing, localeHtmlLang, localeLabels, localeNames } from '@/i18n/routing'

// Language menu. This was a two-button EN / 中文 toggle; with fourteen locales a
// row of buttons no longer fits a navbar, so the trigger collapses to the
// current language and the rest live in a dropdown.
//
// Each option carries its own `lang`, so the browser picks a font that can
// actually draw it: without that, 日本語 and 한국어 can fall back to a font
// with no glyphs for them and render as boxes.
export default function LanguageSwitcher({ className = '' }) {
  const locale = useLocale()
  const pathname = usePathname()
  const router = useRouter()
  const t = useTranslations('LangSwitcher')
  const [isPending, startTransition] = useTransition()
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)
  const triggerRef = useRef(null)

  // Close on outside click or Escape. Escape returns focus to the trigger so
  // keyboard users are not dropped at the top of the document.
  useEffect(() => {
    if (!open) return
    const onPointerDown = (e) => {
      if (!rootRef.current?.contains(e.target)) setOpen(false)
    }
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const switchTo = (next) => {
    setOpen(false)
    if (next === locale) return
    startTransition(() => {
      router.replace(pathname, { locale: next })
    })
  }

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={isPending}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t('label')}
        className="inline-flex items-center gap-1.5 rounded-lg border border-surface-200 px-2 py-1.5 text-xs font-medium text-surface-500 transition-colors hover:text-surface-900 dark:border-white/10 dark:text-surface-400 dark:hover:text-white"
      >
        <Globe className="h-3.5 w-3.5" aria-hidden="true" />
        <span lang={localeHtmlLang[locale]}>{localeLabels[locale]}</span>
      </button>

      {open && (
        <div
          role="menu"
          aria-label={t('label')}
          className="absolute right-0 z-50 mt-2 max-h-80 w-48 overflow-y-auto rounded-xl border border-surface-200 bg-white py-1 shadow-lg dark:border-white/10 dark:bg-surface-900"
        >
          {routing.locales.map((code) => {
            const isActive = code === locale
            return (
              <button
                key={code}
                type="button"
                role="menuitemradio"
                aria-checked={isActive}
                onClick={() => switchTo(code)}
                disabled={isPending}
                lang={localeHtmlLang[code]}
                className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm transition-colors ${
                  isActive
                    ? 'text-libre-600 dark:text-libre-400'
                    : 'text-surface-600 hover:bg-surface-50 dark:text-surface-300 dark:hover:bg-white/5'
                }`}
              >
                <span>{localeNames[code]}</span>
                {isActive && <Check className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
