import { defineRouting } from 'next-intl/routing'

export const routing = defineRouting({
  // English stays at the root (`/models`); every other locale is served under
  // its prefix (`/zh/models`, `/es/models`).
  // Locales roll out page by page: a docs page with a `<slug>.<locale>.md`
  // twin serves translated, and one without it serves English under the
  // English canonical, so a partially translated locale is a valid state.
  // `ar` is deliberately absent until it has pages and the RTL pass is done.
  locales: ['en', 'zh', 'es', 'it', 'pt', 'fr', 'ru', 'id', 'vi', 'de', 'pl', 'ja', 'ko', 'uk'],
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
  id: 'id',
  vi: 'vi',
  de: 'de',
  pl: 'pl',
  ja: 'ja-JP',
  ko: 'ko-KR',
  uk: 'uk',
}

// Full autonyms, listed in the language menu. Each is the language's own name
// for itself, which is what a reader scanning for their language looks for.
export const localeNames = {
  en: 'English',
  zh: '中文',
  es: 'Español',
  it: 'Italiano',
  pt: 'Português',
  fr: 'Français',
  ru: 'Русский',
  id: 'Bahasa Indonesia',
  vi: 'Tiếng Việt',
  de: 'Deutsch',
  pl: 'Polski',
  ja: '日本語',
  ko: '한국어',
  uk: 'Українська',
}

// Flags shown beside each language in the menu.
//
// A flag is a country, not a language, so these are decoration next to the
// autonym and never the only way to identify an option. Where our locale
// already targets a specific variant the choice is factual: `pt` is pt-BR so
// it flies Brazil, `zh` is zh-CN. Two are genuinely arbitrary: `en` and `es`,
// which we write as neutral international Spanish rather than Spain's.
export const localeFlags = {
  en: '🇬🇧',
  zh: '🇨🇳',
  es: '🇪🇸',
  it: '🇮🇹',
  pt: '🇧🇷',
  fr: '🇫🇷',
  ru: '🇷🇺',
  id: '🇮🇩',
  vi: '🇻🇳',
  de: '🇩🇪',
  pl: '🇵🇱',
  ja: '🇯🇵',
  ko: '🇰🇷',
  uk: '🇺🇦',
}

// Short labels for the collapsed trigger, where there is no room for a name.
export const localeLabels = {
  en: 'EN',
  zh: '中文',
  es: 'ES',
  it: 'IT',
  pt: 'PT',
  fr: 'FR',
  ru: 'RU',
  id: 'ID',
  vi: 'VI',
  de: 'DE',
  pl: 'PL',
  ja: '日本語',
  ko: '한국어',
  uk: 'УКР',
}
