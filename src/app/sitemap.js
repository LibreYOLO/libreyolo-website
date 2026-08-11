import { getAllArticles, getArticleBySlug } from '@/lib/articles'
import { routing, localeHtmlLang } from '@/i18n/routing'
import { localeUrl } from '@/i18n/metadata'
import { getAllDocPages, getDoc, DOCS_SECTION_INDEXES } from '@/lib/docs'

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
    { path: '/docs/librevlm', priority: 0.8 },
    { path: '/docs/experimental', priority: 0.8 },
    { path: '/benchmarks', priority: 0.9 },
    { path: '/cursor-hackathon', priority: 0.4 },
  ].flatMap(({ path, priority }) => bilingual(path, priority))

  /*
   * The v2 docs tree, generated from the content directory.
   *
   * Enumerating 169 pages by hand is how a sitemap silently goes stale, so this
   * reads the same manifest the nav, the markdown twins and llms.txt read.
   * Section indexes are message-driven in every locale. Markdown-backed pages
   * only advertise the locales that have a translated twin; untranslated locale
   * fallbacks canonicalize to English and therefore stay out of the sitemap.
   *
   * The frozen v1.1 to v1.4 single-page docs are deliberately absent. They stay
   * reachable and carry a canonical pointing at /docs, and a canonicalised page
   * does not belong in a sitemap.
   */
  const docsIndexRoutes = DOCS_SECTION_INDEXES.flatMap((path) =>
    bilingual(path, path === '/docs' ? 1.0 : 0.8))

  const docsContentRoutes = getAllDocPages().flatMap((page) => {
    const locales = routing.locales.filter((locale) =>
      locale === routing.defaultLocale || getDoc(page.section, page.slug, locale)?.translated
    )
    const priority = page.section === 'models' || page.section === 'tasks' ? 0.8 : 0.7
    return locales.length > 1
      ? localized(page.path, locales, priority, 'weekly', page.lastModified)
      : englishOnly(page.path, priority, 'weekly', page.lastModified)
  })

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

  return [...bilingualRoutes, ...docsIndexRoutes, ...docsContentRoutes, ...articleRoutes]
}
