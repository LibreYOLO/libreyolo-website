import { getAllArticles } from '@/lib/articles'

const BASE_URL = 'https://www.libreyolo.com'

export default function sitemap() {
  const staticRoutes = [
    { path: '', priority: 1.0 },
    { path: '/models', priority: 0.9 },
    { path: '/docs', priority: 0.9 },
    { path: '/docs/librevlm', priority: 0.8 },
    { path: '/docs/experimental', priority: 0.8 },
    { path: '/docs/v1.1.0', priority: 0.5 },
    { path: '/commercial', priority: 0.8 },
    { path: '/science', priority: 0.8 },
    { path: '/articles', priority: 0.9 },
    { path: '/cursor-hackathon', priority: 0.4 },
  ].map(({ path, priority }) => ({
    url: `${BASE_URL}${path}`,
    lastModified: new Date(),
    changeFrequency: 'weekly',
    priority,
  }))

  const articleRoutes = getAllArticles().map((article) => ({
    url: `${BASE_URL}/articles/${article.slug}`,
    lastModified: new Date(article.date),
    changeFrequency: 'monthly',
    priority: 0.7,
  }))

  return [...staticRoutes, ...articleRoutes]
}
