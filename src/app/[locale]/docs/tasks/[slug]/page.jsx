import { notFound } from 'next/navigation'
import { setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs, extractHeadings, DOCS_NAV, DOCS_VERSION } from '@/lib/docs'
import { buildPageMetadata, localeUrl, SITE_URL } from '@/i18n/metadata'
import { routing } from '@/i18n/routing'
import DocsShell from '@/components/docs/DocsShell'
import DocMarkdown from '@/components/docs/DocMarkdown'
import { PageHeader } from '@/components/docs/ModelBlocks'

// Task pages are capability hubs. They carry no single registry family (a task is served by many), so the header rows come from frontmatter and the model list is a generated block in the body.
const SECTION = 'tasks'
const SECTION_LABEL = 'Tasks'

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

  const doc = getDoc(SECTION, slug, locale)
  if (!doc) notFound()

  const path = `/docs/${SECTION}/${slug}`
  const url = localeUrl(path, doc.translated ? locale : routing.defaultLocale)
  const headings = extractHeadings(doc.content)

  const breadcrumbs = [
    { label: 'Docs', href: '/docs' },
    { label: SECTION_LABEL, href: `/docs/${SECTION}` },
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
        nav={DOCS_NAV}
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
              {doc.verification ||
                `Verified against LibreYOLO v${doc.last_verified}.`}
            </p>
          </footer>
        </article>
      </DocsShell>
    </>
  )
}