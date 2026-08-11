import { routing, localeHtmlLang } from './routing'

// Production serves on www (libreyolo.com 307-redirects to www.libreyolo.com),
// so every canonical / hreflang / OG / sitemap URL points at the non-redirecting host.
export const SITE_URL = 'https://www.libreyolo.com'

// The dynamic OG image lives at app/opengraph-image.jsx (served at /opengraph-image).
// Because all pages moved under [locale], the file-convention image is no longer
// auto-attached, so we reference it explicitly in metadata.
export const OG_IMAGE = {
  url: '/opengraph-image',
  width: 1200,
  height: 630,
  alt: 'LibreYOLO: MIT-Licensed Object Detection',
}

// OpenGraph locale string per app locale.
const OG_LOCALES = {
  en: 'en_US',
  zh: 'zh_CN',
  es: 'es_ES',
  it: 'it_IT',
  fr: 'fr_FR',
  ru: 'ru_RU',
  pt: 'pt_BR',
  id: 'id_ID',
  de: 'de_DE',
  ja: 'ja_JP',
  ko: 'ko_KR',
  pl: 'pl_PL',
  uk: 'uk_UA',
  vi: 'vi_VN',
}
export function ogLocale(locale) {
  return OG_LOCALES[locale] ?? OG_LOCALES[routing.defaultLocale]
}

// Absolute URL for a path in a given locale. English (the default) lives at the
// root (`/models`); every other locale is prefixed (`/zh/models`).
export function localeUrl(path, locale) {
  const clean = path === '/' ? '' : path
  if (locale === routing.defaultLocale) return `${SITE_URL}${clean || '/'}`
  return `${SITE_URL}/${locale}${clean}`
}

// hreflang map covering every configured locale + x-default. Derived from
// routing.locales / localeHtmlLang so adding a locale needs no edits here.
function languageMap(path) {
  const languages = {}
  for (const locale of routing.locales) {
    languages[localeHtmlLang[locale]] = localeUrl(path, locale)
  }
  languages['x-default'] = localeUrl(path, routing.defaultLocale)
  return languages
}

// Canonical (self-referential per locale) + full hreflang map for a translated page.
export function buildAlternates(path, locale) {
  return {
    canonical: localeUrl(path, locale),
    languages: languageMap(path),
  }
}

// For genuinely English-only pages: when a /zh/* URL serves the same English
// content, consolidate it to the English canonical instead of indexing a duplicate.
export function buildEnglishOnlyAlternates(path) {
  const url = localeUrl(path, routing.defaultLocale)
  const lang = localeHtmlLang[routing.defaultLocale]
  return {
    canonical: url,
    languages: { [lang]: url, 'x-default': url },
  }
}

// Full per-page metadata (title, description, alternates, OpenGraph, Twitter).
// Setting OpenGraph here means each page advertises its own title/url/card
// instead of inheriting the home page's values from the root layout.
/*
 * `ownImage` opts a route out of the shared social card.
 *
 * Next injects a segment's own `opengraph-image` file automatically, but ONLY
 * when metadata does not set `openGraph.images` explicitly. Setting it here
 * unconditionally is what kept all 171 docs pages sharing one image after the
 * per-page cards existed: the card rendered fine at its URL and nothing
 * pointed at it.
 *
 * The URL is built here rather than left to that auto-injection, which emits
 * a locale-prefixed path (`/en/docs/...`). English is served at the root, so
 * that path answers 307 rather than 200, and several social scrapers do not
 * follow a redirect for `og:image`. `localeUrl` already resolves the default
 * locale to the unprefixed form, so the card URL 200s for every scraper.
 */
function ownImageUrl(path, locale) {
  return `${localeUrl(path, locale)}/opengraph-image`
}

export function buildPageMetadata({ title, description, path, locale, englishOnly = false, ownImage = false }) {
  const ogTarget = englishOnly ? routing.defaultLocale : locale
  return {
    title,
    description,
    alternates: englishOnly ? buildEnglishOnlyAlternates(path) : buildAlternates(path, locale),
    openGraph: {
      title,
      description,
      url: localeUrl(path, ogTarget),
      siteName: 'LibreYOLO',
      locale: ogLocale(ogTarget),
      type: 'website',
      images: ownImage
        ? [{ url: ownImageUrl(path, ogTarget), width: 1200, height: 630, alt: title }]
        : [OG_IMAGE],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [ownImage ? ownImageUrl(path, ogTarget) : OG_IMAGE.url],
    },
  }
}

export { localeHtmlLang }
