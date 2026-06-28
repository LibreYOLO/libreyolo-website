import { getAllArticles } from '@/lib/articles'
import { routing, localeHtmlLang } from '@/i18n/routing'
import { localeUrl } from '@/i18n/metadata'

function languageMap(path) {
  const languages = {}
  for (const locale of routing.locales) {
    languages[localeHtmlLang[locale]] = localeUrl(path, locale)
  }
  return languages
}

// Translated pages: one entry per locale, each carrying the full hreflang map so
// search engines can pair the localized versions.
function bilingual(path, priority, changeFrequency = 'weekly', lastModified = new Date()) {
  const languages = languageMap(path)
  return routing.locales.map((locale) => ({
    url: localeUrl(path, locale),
    lastModified,
    changeFrequency,
    priority,
    alternates: { languages },
  }))
}

// English-only pages (docs, hackathon, individual articles): a single English entry.
function englishOnly(path, priority, changeFrequency = 'weekly', lastModified = new Date()) {
  return [{ url: localeUrl(path, routing.defaultLocale), lastModified, changeFrequency, priority }]
}

export default function sitemap() {
  const bilingualRoutes = [
    { path: '', priority: 1.0 },
    { path: '/models', priority: 0.9 },
    { path: '/commercial', priority: 0.8 },
    { path: '/science', priority: 0.8 },
    { path: '/datasets', priority: 0.7 },
    { path: '/articles', priority: 0.9 },
  ].flatMap(({ path, priority }) => bilingual(path, priority))

  const englishOnlyRoutes = [
    { path: '/docs', priority: 0.9 },
    { path: '/docs/v1.3.0', priority: 0.7 },
    { path: '/docs/librevlm', priority: 0.8 },
    { path: '/docs/experimental', priority: 0.8 },
    { path: '/docs/v1.1.0', priority: 0.5 },
    { path: '/cursor-hackathon', priority: 0.4 },
  ].flatMap(({ path, priority }) => englishOnly(path, priority))

  const articleRoutes = getAllArticles().flatMap((article) =>
    englishOnly(`/articles/${article.slug}`, 0.7, 'monthly', new Date(article.date))
  )

  return [...bilingualRoutes, ...englishOnlyRoutes, ...articleRoutes]
}
