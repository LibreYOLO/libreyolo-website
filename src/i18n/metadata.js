import { localeHtmlLang } from './routing'

export const SITE_URL = 'https://libreyolo.com'

// The dynamic OG image lives at app/opengraph-image.jsx (served at /opengraph-image).
// Because all pages moved under [locale], the file-convention image is no longer
// auto-attached, so we reference it explicitly in metadata.
export const OG_IMAGE = {
  url: '/opengraph-image',
  width: 1200,
  height: 630,
  alt: 'LibreYOLO: MIT-Licensed Object Detection',
}

// Builds canonical + hreflang alternates for a route, given the locale-agnostic
// path (e.g. '/models', or '/' for home). English lives at the root, Chinese
// under '/zh'. The canonical is self-referential per locale; the language map is
// identical across locales so search engines can pair them up.
export function buildAlternates(path, locale) {
  const clean = path === '/' ? '' : path
  const enUrl = `${SITE_URL}${clean || '/'}`
  const zhUrl = `${SITE_URL}/zh${clean}`

  return {
    canonical: locale === 'zh' ? zhUrl : enUrl,
    languages: {
      en: enUrl,
      'zh-CN': zhUrl,
      'x-default': enUrl,
    },
  }
}

// OpenGraph locale string for a given app locale.
export function ogLocale(locale) {
  return locale === 'zh' ? 'zh_CN' : 'en_US'
}

export { localeHtmlLang }
