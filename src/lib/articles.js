import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const articlesDir = path.join(process.cwd(), 'content', 'articles')

export function getAllArticles() {
  if (!fs.existsSync(articlesDir)) return []

  return fs
    .readdirSync(articlesDir)
    .filter((file) => file.endsWith('.md'))
    .map((file) => {
      const slug = file.replace(/\.md$/, '')
      const raw = fs.readFileSync(path.join(articlesDir, file), 'utf8')
      const { data, content } = matter(raw)
      return {
        slug,
        title: data.title || slug,
        description: data.description || '',
        date: data.date || '1970-01-01',
        author: data.author || 'LibreYOLO Team',
        tags: data.tags || [],
        content,
      }
    })
    .sort((a, b) => new Date(b.date) - new Date(a.date))
}

export function getArticleBySlug(slug) {
  return getAllArticles().find((article) => article.slug === slug) || null
}
