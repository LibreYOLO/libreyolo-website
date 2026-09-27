import { getAllArticles, getArticleBySlug } from '@/lib/articles'
import { routing, localeHtmlLang } from '@/i18n/routing'
import { localeUrl } from '@/i18n/metadata'
import { getAllDocPages, getDoc, DOCS_SECTION_INDEXES } from '@/lib/docs'
import { contentDate, localeFile } from '@/lib/content-dates'

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
// `lastModified` is a date, or a function of the locale when each translation
// has its own file and therefore its own last change.
function localized(path, locales, priority, changeFrequency = 'weekly', lastModified) {
  const languages = languageMap(path, locales)
  return locales.map((locale) => ({
    url: localeUrl(path, locale),
    ...dateOf(lastModified, locale),
    changeFrequency,
    priority,
    alternates: { languages },
  }))
}

function bilingual(path, priority, changeFrequency = 'weekly', lastModified) {
  return localized(path, routing.locales, priority, changeFrequency, lastModified)
}

function dateOf(lastModified, locale) {
  const date = typeof lastModified === 'function' ? lastModified(locale) : lastModified
  return date ? { lastModified: date } : {}
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
    { path: '/sponsors', priority: 0.6 },
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
   * The 1.6.0 model, task, workflow and API pages are included automatically.
   * Enumerating docs pages by hand is how a sitemap silently goes stale, so this
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
    const base = `content/docs/${page.section}/${page.slug}`
    const lastModified = (locale) => contentDate(localeFile(base, locale))
    return locales.length > 1
      ? localized(page.path, locales, priority, 'weekly', lastModified)
      : englishOnly(page.path, priority, 'weekly', lastModified('en'))
  })

  const articleRoutes = getAllArticles().flatMap((article) => {
    const locales = routing.locales.filter((locale) =>
      locale === routing.defaultLocale || getArticleBySlug(article.slug, locale)?.translated
    )
    const path = `/articles/${article.slug}`
    const base = `content/articles/${article.slug}`
    const lastModified = (locale) =>
      contentDate(localeFile(base, locale)) ?? new Date(article.date)
    return locales.length > 1
      ? localized(path, locales, 0.7, 'monthly', lastModified)
      : englishOnly(path, 0.7, 'monthly', lastModified('en'))
  })

  return [...bilingualRoutes, ...docsIndexRoutes, ...docsContentRoutes, ...articleRoutes]
}
