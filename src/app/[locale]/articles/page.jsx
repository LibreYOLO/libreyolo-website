import Link from 'next/link'
import { getAllArticles } from '@/lib/articles'

export const metadata = {
  title: 'Articles',
  description: 'Articles, tutorials, and news about LibreYOLO: MIT-licensed object detection, training tips, model releases, and computer vision guides.',
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  })
}

function readingTime(content = '') {
  const words = content.trim().split(/\s+/).filter(Boolean).length
  return Math.max(1, Math.round(words / 200))
}

export default function Articles() {
  const articles = getAllArticles()

  return (
    <div className="pt-28 lg:pt-36 pb-24">
      <div className="max-w-2xl mx-auto px-6">
        {/* Header */}
        <header className="mb-14 lg:mb-16">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-surface-900 dark:text-white mb-4">
            Articles
          </h1>
          <p className="text-lg text-surface-500 dark:text-surface-400">
            Tutorials and deep dives on running detection models with LibreYOLO.
          </p>
        </header>

        {/* Editorial list */}
        {articles.length === 0 ? (
          <p className="text-surface-500">No articles yet. Check back soon.</p>
        ) : (
          <div className="divide-y divide-surface-200 dark:divide-white/10 border-t border-surface-200 dark:border-white/10">
            {articles.map((article) => (
              <article key={article.slug}>
                <Link href={`/articles/${article.slug}`} className="group block py-8">
                  <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-surface-900 dark:text-white group-hover:text-libre-600 dark:group-hover:text-libre-400 transition-colors">
                    {article.title}
                  </h2>
                  <div className="mt-2 text-sm text-surface-400 dark:text-surface-500 font-mono">
                    {formatDate(article.date)} &middot; {readingTime(article.content)} min read
                  </div>
                  <p className="mt-3 text-surface-600 dark:text-surface-400 leading-relaxed">
                    {article.description}
                  </p>
                </Link>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
