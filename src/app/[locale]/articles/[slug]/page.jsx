import { notFound } from 'next/navigation'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { ArrowLeft, Calendar, User } from 'lucide-react'
import { getTranslations, setRequestLocale } from 'next-intl/server'
import { getAllArticles, getArticleBySlug } from '@/lib/articles'
import {
  buildAlternates,
  buildEnglishOnlyAlternates,
  localeUrl,
  ogLocale,
  localeHtmlLang,
  OG_IMAGE,
  SITE_URL,
} from '@/i18n/metadata'
import { routing } from '@/i18n/routing'
import { Link } from '@/i18n/navigation'
import ThemedEmbed from '@/components/ThemedEmbed'
import RF100VLHero from '@/components/articles/rf100vl/RF100VLHero'
import RF100VLResults from '@/components/articles/rf100vl/RF100VLResults'

export function generateStaticParams() {
  return getAllArticles().map((article) => ({ slug: article.slug }))
}

export async function generateMetadata({ params }) {
  const { locale, slug } = await params
  const article = getArticleBySlug(slug, locale)
  if (!article) return {}

  const path = `/articles/${article.slug}`
  // English source articles report `translated: false`, so also check whether a
  // localized counterpart exists. Both sides of a translated pair must publish
  // the same reciprocal hreflang set. A locale fallback that serves English still
  // consolidates to the English canonical to avoid duplicate indexing.
  const hasLocalizedCounterpart = routing.locales.some((candidateLocale) =>
    candidateLocale !== routing.defaultLocale &&
    getArticleBySlug(slug, candidateLocale)?.translated
  )
  const isLocalizedVersion = locale === routing.defaultLocale
    ? hasLocalizedCounterpart
    : article.translated
  const ogTarget = isLocalizedVersion ? locale : routing.defaultLocale

  return {
    title: article.title,
    description: article.description,
    keywords: article.tags,
    alternates: isLocalizedVersion
      ? buildAlternates(path, locale)
      : buildEnglishOnlyAlternates(path),
    openGraph: {
      title: article.title,
      description: article.description,
      url: localeUrl(path, ogTarget),
      siteName: 'LibreYOLO',
      locale: ogLocale(ogTarget),
      type: 'article',
      publishedTime: article.date,
      authors: [article.author],
      images: [OG_IMAGE],
    },
    twitter: {
      card: 'summary_large_image',
      title: article.title,
      description: article.description,
      images: [OG_IMAGE.url],
    },
  }
}

function formatDate(dateString, locale) {
  return new Date(dateString).toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US', {
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
  iframe: ThemedEmbed,
  // Custom widgets an article can drop inline via raw-HTML tags in the markdown,
  // e.g. <rf100vl-hero />. rehype-raw keeps unknown tags, so they land here.
  'rf100vl-hero': () => <RF100VLHero />,
  'rf100vl-results': () => <RF100VLResults />,
}

export default async function ArticlePage({ params }) {
  const { locale, slug } = await params
  setRequestLocale(locale)
  const article = getArticleBySlug(slug, locale)
  if (!article) notFound()
  const t = await getTranslations({ locale, namespace: 'Articles' })

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: article.title,
    description: article.description,
    datePublished: article.date,
    author: { '@type': 'Person', name: article.author },
    publisher: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
    inLanguage: localeHtmlLang[article.translated ? locale : routing.defaultLocale],
    mainEntityOfPage: localeUrl(`/articles/${article.slug}`, article.translated ? locale : routing.defaultLocale),
  }

  // When the article declares an `faq` list in frontmatter, emit FAQPage schema
  // so the Q&A is eligible for rich results and clean LLM extraction. The marked-up
  // text mirrors the visible FAQ section, as Google requires.
  const faqJsonLd =
    article.faq && article.faq.length
      ? {
          '@context': 'https://schema.org',
          '@type': 'FAQPage',
          mainEntity: article.faq.map((item) => ({
            '@type': 'Question',
            name: item.q,
            acceptedAnswer: { '@type': 'Answer', text: item.a },
          })),
        }
      : null

  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {faqJsonLd && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
        />
      )}
      <article className="max-w-3xl mx-auto px-6 lg:px-8">
        <Link
          href="/articles"
          className="inline-flex items-center gap-1.5 text-sm text-surface-500 hover:text-libre-600 dark:hover:text-libre-400 transition-colors mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          {t('backToArticles')}
        </Link>

        <header className="mb-10">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-800 dark:text-white mb-4 leading-tight">
            {article.title}
          </h1>
          <div className="flex flex-wrap items-center gap-4 text-sm text-surface-500">
            <span className="inline-flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <time dateTime={article.date}>{formatDate(article.date, locale)}</time>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <User className="w-3.5 h-3.5" />
              {article.author}
            </span>
          </div>
          {locale === 'zh' && !article.translated && (
            <p className="mt-4 text-sm text-surface-400 dark:text-surface-500">
              {t('englishNote')}
            </p>
          )}
        </header>

        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={markdownComponents}>
          {article.content}
        </ReactMarkdown>
      </article>
    </div>
  )
}
