import { notFound } from 'next/navigation'
import { getTranslations, setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs, extractHeadings, DOCS_VERSION, localizeNav } from '@/lib/docs'
import { buildPageMetadata, localeUrl, SITE_URL } from '@/i18n/metadata'
import { routing } from '@/i18n/routing'
import DocsShell from '@/components/docs/DocsShell'
import DocMarkdown from '@/components/docs/DocMarkdown'
import { PageHeader } from '@/components/docs/ModelBlocks'

// One page per command. The synopsis and argument table live in the markdown body; the header rows carry the command and its one-line purpose.
const SECTION = 'cli'

export function generateStaticParams() {
  return getDocSlugs(SECTION).map((slug) => ({ slug }))
}

export async function generateMetadata({ params }) {
  const { locale, slug } = await params
  const doc = getDoc(SECTION, slug, locale)
  if (!doc) return {}

  const path = `/docs/${SECTION}/${slug}`
  return {
    ...buildPageMetadata({
      title: doc.seo_title || doc.title,
      description: doc.description,
      path,
      locale,
      // Until a page has a .zh.md twin, a /zh URL serves English and
      // consolidates to the English canonical rather than claiming a
      // translation that does not exist.
      englishOnly: !doc.translated,
      ownImage: true,
    }),
    keywords: doc.keywords,
  }
}

export default async function DocPage({ params }) {
  const { locale, slug } = await params
  setRequestLocale(locale)
  const t = await getTranslations({ locale, namespace: 'DocsChrome' })

  const doc = getDoc(SECTION, slug, locale)
  if (!doc) notFound()

  const path = `/docs/${SECTION}/${slug}`
  const url = localeUrl(path, doc.translated ? locale : routing.defaultLocale)
  const headings = extractHeadings(doc.content)

  const breadcrumbs = [
    { label: t('docsCrumb'), href: '/docs' },
    { label: t(`groups.${SECTION}`), href: `/docs/${SECTION}` },
    { label: doc.title },
  ]

  const jsonLd = [
    {
      '@context': 'https://schema.org',
      '@type': 'TechArticle',
      headline: doc.seo_title || doc.title,
      description: doc.description,
      mainEntityOfPage: url,
      inLanguage: doc.translated ? locale : routing.defaultLocale,
      publisher: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
      about: doc.title,
      proficiencyLevel: 'Beginner',
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: breadcrumbs.map((crumb, index) => ({
        '@type': 'ListItem',
        position: index + 1,
        name: crumb.label,
        ...(crumb.href ? { item: `${SITE_URL}${crumb.href}` } : {}),
      })),
    },
  ]

  return (
    <>
      {jsonLd.map((block, index) => (
        <script
          key={index}
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(block) }}
        />
      ))}

      <DocsShell
        nav={localizeNav(locale)}
        activePath={path}
        version={DOCS_VERSION}
        headings={headings}
        breadcrumbs={breadcrumbs}
      >
        <article className="max-w-3xl">
          <PageHeader doc={doc} />

          <DocMarkdown snippets={doc.snippets || {}} bareTables>
            {doc.content}
          </DocMarkdown>

          <footer className="mt-16 border-t border-surface-200 pt-6 text-sm text-surface-500 dark:border-white/[0.06] dark:text-surface-500">
            <p>
              {doc.verification || t('verified', { version: doc.last_verified })}
            </p>
          </footer>
        </article>
      </DocsShell>
    </>
  )
}
