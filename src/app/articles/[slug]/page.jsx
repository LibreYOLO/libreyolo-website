import Link from 'next/link'
import { notFound } from 'next/navigation'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ArrowLeft, Calendar, User } from 'lucide-react'
import { getAllArticles, getArticleBySlug } from '@/lib/articles'

export function generateStaticParams() {
  return getAllArticles().map((article) => ({ slug: article.slug }))
}

export async function generateMetadata({ params }) {
  const { slug } = await params
  const article = getArticleBySlug(slug)
  if (!article) return {}

  return {
    title: article.title,
    description: article.description,
    keywords: article.tags,
    alternates: {
      canonical: `/articles/${article.slug}`,
    },
    openGraph: {
      title: article.title,
      description: article.description,
      url: `https://libreyolo.com/articles/${article.slug}`,
      siteName: 'LibreYOLO',
      type: 'article',
      publishedTime: article.date,
      authors: [article.author],
    },
    twitter: {
      card: 'summary_large_image',
      title: article.title,
      description: article.description,
    },
  }
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

const markdownComponents = {
  h1: (props) => <h2 className="text-3xl font-bold text-surface-800 dark:text-white mt-12 mb-4" {...props} />,
  h2: (props) => <h2 className="text-2xl font-bold text-surface-800 dark:text-white mt-12 mb-4" {...props} />,
  h3: (props) => <h3 className="text-xl font-semibold text-surface-800 dark:text-white mt-8 mb-3" {...props} />,
  p: (props) => <p className="text-surface-600 dark:text-surface-400 leading-relaxed mb-5" {...props} />,
  a: (props) => (
    <a
      className="text-libre-600 dark:text-libre-400 hover:text-libre-700 dark:hover:text-libre-300 underline underline-offset-2 transition-colors"
      target={props.href?.startsWith('http') ? '_blank' : undefined}
      rel={props.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
      {...props}
    />
  ),
  ul: (props) => <ul className="list-disc pl-6 mb-5 space-y-2 text-surface-600 dark:text-surface-400" {...props} />,
  ol: (props) => <ol className="list-decimal pl-6 mb-5 space-y-2 text-surface-600 dark:text-surface-400" {...props} />,
  li: (props) => <li className="leading-relaxed" {...props} />,
  code: ({ className, children, ...props }) => {
    const isBlock = /language-/.test(className || '')
    if (isBlock) {
      return (
        <code className={`${className} block font-mono text-sm`} {...props}>
          {children}
        </code>
      )
    }
    return (
      <code className="font-mono text-sm px-1.5 py-0.5 rounded bg-surface-100 dark:bg-surface-800 text-libre-700 dark:text-libre-300" {...props}>
        {children}
      </code>
    )
  },
  pre: (props) => (
    <pre className="rounded-xl border border-surface-200 dark:border-surface-800 bg-surface-50 dark:bg-surface-900 p-5 overflow-x-auto mb-6 text-surface-800 dark:text-surface-200" {...props} />
  ),
  blockquote: (props) => (
    <blockquote className="border-l-4 border-libre-500 pl-4 italic text-surface-500 dark:text-surface-400 my-6" {...props} />
  ),
  table: (props) => (
    <div className="overflow-x-auto mb-6">
      <table className="w-full text-sm border-collapse" {...props} />
    </div>
  ),
  th: (props) => <th className="text-left font-semibold text-surface-800 dark:text-white border-b border-surface-300 dark:border-surface-700 px-3 py-2" {...props} />,
  td: (props) => <td className="text-surface-600 dark:text-surface-400 border-b border-surface-200 dark:border-surface-800 px-3 py-2" {...props} />,
  img: (props) => <img className="rounded-xl my-6 mx-auto" loading="lazy" {...props} />,
  hr: () => <hr className="border-surface-200 dark:border-surface-800 my-10" />,
}

export default async function ArticlePage({ params }) {
  const { slug } = await params
  const article = getArticleBySlug(slug)
  if (!article) notFound()

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: article.title,
    description: article.description,
    datePublished: article.date,
    author: { '@type': 'Person', name: article.author },
    publisher: { '@type': 'Organization', name: 'LibreYOLO', url: 'https://libreyolo.com' },
    mainEntityOfPage: `https://libreyolo.com/articles/${article.slug}`,
  }

  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <article className="max-w-3xl mx-auto px-6 lg:px-8">
        <Link
          href="/articles"
          className="inline-flex items-center gap-1.5 text-sm text-surface-500 hover:text-libre-600 dark:hover:text-libre-400 transition-colors mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          All articles
        </Link>

        <header className="mb-10">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-800 dark:text-white mb-4 leading-tight">
            {article.title}
          </h1>
          <div className="flex flex-wrap items-center gap-4 text-sm text-surface-500">
            <span className="inline-flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <time dateTime={article.date}>{formatDate(article.date)}</time>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <User className="w-3.5 h-3.5" />
              {article.author}
            </span>
          </div>
        </header>

        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {article.content}
        </ReactMarkdown>
      </article>
    </div>
  )
}
