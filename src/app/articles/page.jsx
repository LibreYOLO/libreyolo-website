import Link from 'next/link'
import { Newspaper, ArrowRight, Calendar, User } from 'lucide-react'
import { getAllArticles } from '@/lib/articles'

export const metadata = {
  title: 'Articles',
  description: 'Articles, tutorials, and news about LibreYOLO: MIT-licensed object detection, training tips, model releases, and computer vision guides.',
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export default function Articles() {
  const articles = getAllArticles()

  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-4xl mx-auto px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-700 dark:text-libre-400 text-sm font-medium mb-6">
            <Newspaper className="w-4 h-4" />
            Blog
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            <span className="text-libre-600 dark:text-libre-400">Articles</span>
          </h1>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            Tutorials, release notes, and deep dives on object detection with LibreYOLO.
          </p>
        </div>

        {/* Article list */}
        {articles.length === 0 ? (
          <p className="text-center text-surface-500">No articles yet. Check back soon!</p>
        ) : (
          <div className="space-y-6">
            {articles.map((article) => (
              <Link
                key={article.slug}
                href={`/articles/${article.slug}`}
                className="group block p-6 sm:p-8 rounded-2xl border border-surface-200 dark:border-surface-800 bg-white dark:bg-surface-900/50 hover:border-libre-500/50 dark:hover:border-libre-400/50 transition-colors"
              >
                <div className="flex items-center gap-4 text-sm text-surface-500 mb-3">
                  <span className="inline-flex items-center gap-1.5">
                    <Calendar className="w-3.5 h-3.5" />
                    {formatDate(article.date)}
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <User className="w-3.5 h-3.5" />
                    {article.author}
                  </span>
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-surface-800 dark:text-white mb-2 group-hover:text-libre-600 dark:group-hover:text-libre-400 transition-colors">
                  {article.title}
                </h2>
                <p className="text-surface-600 dark:text-surface-400 mb-4">
                  {article.description}
                </p>
                <span className="inline-flex items-center gap-1.5 text-sm font-medium text-libre-600 dark:text-libre-400">
                  Read article
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
