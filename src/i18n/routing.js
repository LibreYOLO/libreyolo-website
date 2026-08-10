import { defineRouting } from 'next-intl/routing'

export const routing = defineRouting({
  // English stays at the root (`/models`); every other locale is served under
  // its prefix (`/zh/models`, `/es/models`).
  // Locales roll out page by page: a docs page with a `<slug>.<locale>.md`
  // twin serves translated, and one without it serves English under the
  // English canonical, so a partially translated locale is a valid state.
  // `ar` is deliberately absent until it has pages and the RTL pass is done.
  locales: ['en', 'zh', 'es', 'it', 'pt', 'fr', 'ru'],
  defaultLocale: 'en',
  localePrefix: 'as-needed',
})

// BCP-47 tags used for the <html lang> attribute and hreflang alternates.
export const localeHtmlLang = {
  en: 'en',
  zh: 'zh-CN',
  es: 'es',
  it: 'it',
  pt: 'pt-BR',
  fr: 'fr',
  ru: 'ru',
}

// Autonym labels shown in the language switcher.
export const localeLabels = {
  en: 'EN',
  zh: '中文',
  es: 'ES',
  it: 'IT',
  pt: 'PT',
  fr: 'FR',
  ru: 'RU',
}
