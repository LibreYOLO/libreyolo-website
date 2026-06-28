import { getAllArticles } from '@/lib/articles'
import { SITE_URL } from '@/i18n/metadata'

// Host kept consistent with the canonical/hreflang URLs emitted in metadata.
const BASE = SITE_URL

function enUrl(path) {
  const clean = path === '/' ? '' : path
  return `${BASE}${clean || '/'}`
}

function zhUrl(path) {
  const clean = path === '/' ? '' : path
  return `${BASE}/zh${clean}`
}

// Translated pages: emit both locales, each carrying the full hreflang map so
// search engines can pair the English and Chinese versions.
function bilingual(path, priority, changeFrequency = 'weekly', lastModified = new Date()) {
  const languages = { en: enUrl(path), 'zh-CN': zhUrl(path) }
  return [
    { url: enUrl(path), lastModified, changeFrequency, priority, alternates: { languages } },
    { url: zhUrl(path), lastModified, changeFrequency, priority, alternates: { languages } },
  ]
}

// English-only pages (docs, hackathon, individual articles): one entry, English URL.
function englishOnly(path, priority, changeFrequency = 'weekly', lastModified = new Date()) {
  return [{ url: enUrl(path), lastModified, changeFrequency, priority }]
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
