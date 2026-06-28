import { defineRouting } from 'next-intl/routing'

export const routing = defineRouting({
  // English stays at the root (`/models`); Chinese is served under `/zh` (`/zh/models`).
  locales: ['en', 'zh'],
  defaultLocale: 'en',
  localePrefix: 'as-needed',
})

// BCP-47 tags used for the <html lang> attribute and hreflang alternates.
export const localeHtmlLang = {
  en: 'en',
  zh: 'zh-CN',
}
