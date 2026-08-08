import { notFound } from 'next/navigation'
import { setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs, getFamilies, extractHeadings, DOCS_NAV, DOCS_VERSION } from '@/lib/docs'
import { buildPageMetadata, localeUrl, SITE_URL } from '@/i18n/metadata'
import { routing } from '@/i18n/routing'
import DocsShell from '@/components/docs/DocsShell'
import DocMarkdown from '@/components/docs/DocMarkdown'
import { ModelHeader, HeroMedia } from '@/components/docs/ModelBlocks'

const SECTION = 'models'

export function generateStaticParams() {
  return getDocSlugs(SECTION).map((slug) => ({ slug }))
}

export async function generateMetadata({ params }) {
  const { locale, slug } = await params
  const doc = getDoc(SECTION, slug, locale)
  if (!doc) return {}

  const path = `/docs/models/${slug}`
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
    }),
    keywords: doc.keywords,
  }
}

export default async function ModelDocPage({ params }) {
  const { locale, slug } = await params
  setRequestLocale(locale)

  const doc = getDoc(SECTION, slug, locale)
  if (!doc) notFound()

  const [family] = getFamilies(doc.families)
  if (!family) notFound()

  const path = `/docs/models/${slug}`
  const url = localeUrl(path, doc.translated ? locale : routing.defaultLocale)
  // Model pages are a usage reference: install, predict, variants, train,
  // validate, export, licensing. No FAQ and no related-links section.
  const headings = extractHeadings(doc.content)

  const breadcrumbs = [
    { label: 'Docs', href: '/docs' },
    { label: 'Models', href: '/docs/models' },
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
      about: family.display,
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
    {
      '@context': 'https://schema.org',
      '@type': 'SoftwareSourceCode',
      name: family.display,
      description: doc.description,
      codeRepository: 'https://github.com/LibreYOLO/libreyolo',
      programmingLanguage: 'Python',
      license: 'https://opensource.org/license/mit',
      runtimePlatform: 'PyTorch',
    },
  ]

  if (doc.hero?.src) {
    jsonLd.push({
      '@context': 'https://schema.org',
      '@type': 'VideoObject',
      name: `${family.display} running in LibreYOLO`,
      description: doc.hero.caption,
      thumbnailUrl: `${SITE_URL}${doc.hero.poster}`,
      contentUrl: `${SITE_URL}${doc.hero.src}`,
      uploadDate: '2026-08-08',
    })
  }

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
          <ModelHeader doc={doc} family={family} />
          <HeroMedia media={doc.hero} />

          <DocMarkdown family={family} snippets={doc.snippets || {}}>
            {doc.content}
          </DocMarkdown>

          <footer className="mt-16 border-t border-surface-200 pt-6 text-sm text-surface-500 dark:border-white/[0.06] dark:text-surface-500">
            <p>
              Verified against LibreYOLO v{doc.last_verified}. Support tables, checkpoints and
              benchmark numbers on this page are generated from the released library and the
              published weights, not written by hand.
            </p>
          </footer>
        </article>
      </DocsShell>
    </>
  )
}
