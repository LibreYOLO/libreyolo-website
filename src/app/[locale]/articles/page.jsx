import { getTranslations, setRequestLocale } from 'next-intl/server'
import { getAllArticles } from '@/lib/articles'
import { buildPageMetadata } from '@/i18n/metadata'
import { Link } from '@/i18n/navigation'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('articlesTitle'),
    description: t('articlesDescription'),
    path: '/articles',
    locale,
  })
}

function formatDate(dateString, locale) {
  return new Date(dateString).toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US', {
    month: 'short',
    day: 'numeric',
  })
}

function readingTime(content = '') {
  const words = content.trim().split(/\s+/).filter(Boolean).length
  return Math.max(1, Math.round(words / 200))
}

export default async function Articles({ params }) {
  const { locale } = await params
  setRequestLocale(locale)
  const t = await getTranslations({ locale, namespace: 'Articles' })
  const articles = getAllArticles(locale)
  const hasUntranslated = articles.some((article) => !article.translated)

  return (
    <div className="pt-28 lg:pt-36 pb-24">
      <div className="max-w-2xl mx-auto px-6">
        {/* Header */}
        <header className="mb-14 lg:mb-16">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-surface-900 dark:text-white mb-4">
            {t('title')}
          </h1>
          <p className="text-lg text-surface-500 dark:text-surface-400">
            {t('subtitle')}
          </p>
          {locale === 'zh' && hasUntranslated && (
            <p className="mt-3 text-sm text-surface-400 dark:text-surface-500">
              {t('englishNote')}
            </p>
          )}
        </header>

        {/* Editorial list */}
        {articles.length === 0 ? (
          <p className="text-surface-500">{t('empty')}</p>
        ) : (
          <div className="divide-y divide-surface-200 dark:divide-white/10 border-t border-surface-200 dark:border-white/10">
            {articles.map((article) => (
              <article key={article.slug}>
                <Link href={`/articles/${article.slug}`} className="group block py-8">
                  <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-surface-900 dark:text-white group-hover:text-libre-600 dark:group-hover:text-libre-400 transition-colors">
                    {article.title}
                  </h2>
                  <div className="mt-2 text-sm text-surface-400 dark:text-surface-500 font-mono">
                    {formatDate(article.date, locale)} &middot; {t('minRead', { minutes: readingTime(article.content) })}
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
