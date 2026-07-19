import { getAllArticles, getArticleBySlug } from '@/lib/articles'
import { routing, localeHtmlLang } from '@/i18n/routing'
import { localeUrl } from '@/i18n/metadata'

function languageMap(path, locales) {
  const languages = {}
  for (const locale of locales) {
    languages[localeHtmlLang[locale]] = localeUrl(path, locale)
  }
  languages['x-default'] = localeUrl(path, routing.defaultLocale)
  return languages
}

// Localized pages: one entry per translated locale, each carrying the same
// reciprocal hreflang map so search engines can pair every available version.
function localized(path, locales, priority, changeFrequency = 'weekly', lastModified) {
  const languages = languageMap(path, locales)
  return locales.map((locale) => ({
    url: localeUrl(path, locale),
    ...(lastModified ? { lastModified } : {}),
    changeFrequency,
    priority,
    alternates: { languages },
  }))
}

function bilingual(path, priority, changeFrequency = 'weekly', lastModified) {
  return localized(path, routing.locales, priority, changeFrequency, lastModified)
}

// Genuinely English-only pages get one canonical sitemap entry.
function englishOnly(path, priority, changeFrequency = 'weekly', lastModified) {
  return [{
    url: localeUrl(path, routing.defaultLocale),
    ...(lastModified ? { lastModified } : {}),
    changeFrequency,
    priority,
  }]
}

export default function sitemap() {
  const bilingualRoutes = [
    { path: '', priority: 1.0 },
    { path: '/models', priority: 0.9 },
    { path: '/commercial', priority: 0.8 },
    { path: '/science', priority: 0.8 },
    { path: '/datasets', priority: 0.7 },
    { path: '/articles', priority: 0.9 },
    { path: '/docs/v1.3.1', priority: 0.9 },
    { path: '/docs/librevlm', priority: 0.8 },
    { path: '/docs/experimental', priority: 0.8 },
    { path: '/docs/v1.3.0', priority: 0.6 },
    { path: '/docs/v1.2.0', priority: 0.5 },
    { path: '/docs/v1.1.0', priority: 0.5 },
  ].flatMap(({ path, priority }) => bilingual(path, priority))

  const englishOnlyRoutes = [
    // `/docs` redirects to the current version and is intentionally omitted.
    { path: '/cursor-hackathon', priority: 0.4 },
  ].flatMap(({ path, priority }) => englishOnly(path, priority))

  const articleRoutes = getAllArticles().flatMap((article) => {
    const locales = routing.locales.filter((locale) =>
      locale === routing.defaultLocale || getArticleBySlug(article.slug, locale)?.translated
    )
    const path = `/articles/${article.slug}`
    const lastModified = new Date(article.date)
    return locales.length > 1
      ? localized(path, locales, 0.7, 'monthly', lastModified)
      : englishOnly(path, 0.7, 'monthly', lastModified)
  })

  return [...bilingualRoutes, ...englishOnlyRoutes, ...articleRoutes]
}
