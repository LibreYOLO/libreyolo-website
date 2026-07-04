import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const articlesDir = path.join(process.cwd(), 'content', 'articles')

// Base (English) articles live at `content/articles/<slug>.md`. A translation
// lives next to it as `<slug>.<locale>.md` (e.g. `<slug>.zh.md`) and is used when
// present; otherwise we fall back to the English file. The `translated` flag lets
// the pages know whether a reader is seeing the original English or a translation.
function readArticle(slug, locale) {
  const englishPath = path.join(articlesDir, `${slug}.md`)
  const localizedPath =
    locale && locale !== 'en' ? path.join(articlesDir, `${slug}.${locale}.md`) : null

  let filePath = englishPath
  let translated = false
  if (localizedPath && fs.existsSync(localizedPath)) {
    filePath = localizedPath
    translated = true
  }
  if (!fs.existsSync(filePath)) return null

  const raw = fs.readFileSync(filePath, 'utf8')
  const { data, content } = matter(raw)
  return {
    slug,
    title: data.title || slug,
    description: data.description || '',
    date: data.date || '1970-01-01',
    author: data.author || 'LibreYOLO Team',
    tags: data.tags || [],
    // Optional FAQ pairs ([{ q, a }]) that drive FAQPage JSON-LD on the article
    // page. Absent on most articles, so default to null and gate rendering on it.
    faq: Array.isArray(data.faq) ? data.faq : null,
    content,
    translated,
  }
}

export function getAllArticles(locale = 'en') {
  if (!fs.existsSync(articlesDir)) return []

  return fs
    .readdirSync(articlesDir)
    // Only the English files define the canonical slug list; skip `<slug>.<locale>.md`.
    .filter((file) => file.endsWith('.md') && !/\.[a-z]{2}\.md$/.test(file))
    .map((file) => readArticle(file.replace(/\.md$/, ''), locale))
    .filter(Boolean)
    .sort((a, b) => new Date(b.date) - new Date(a.date))
}

export function getArticleBySlug(slug, locale = 'en') {
  return readArticle(slug, locale)
}
