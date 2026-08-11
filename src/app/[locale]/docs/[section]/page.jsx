import { notFound } from 'next/navigation'
import { getTranslations, setRequestLocale } from 'next-intl/server'
import Link from 'next/link'

import {
  DOCS_VERSION,
  getTierMeta,
  getDoc,
  getDocSlugs,
  extractHeadings,
  localizeNav,
} from '@/lib/docs'
import { buildPageMetadata, localeUrl, SITE_URL } from '@/i18n/metadata'
import { routing } from '@/i18n/routing'
import DocsShell from '@/components/docs/DocsShell'
import DocMarkdown from '@/components/docs/DocMarkdown'
import { PageHeader } from '@/components/docs/ModelBlocks'

/*
 * Section index: /docs/models, /docs/export, and the rest.
 *
 * These exist because every page below them puts the section in its breadcrumb
 * and in its BreadcrumbList JSON-LD. A breadcrumb pointing at a 404 is a broken
 * promise to the reader and a structured-data error to a crawler.
 *
 * The listing is the nav manifest, so it cannot drift from the sidebar. Pages
 * not yet written are shown as plain text rather than hidden: the reader learns
 * what exists and what is coming, and no link leads nowhere.
 */

const SECTIONS = {
  models: {
    group: 'models',
  },
  tasks: {
    group: 'tasks',
  },
  export: {
    group: 'export',
  },
  train: {
    group: 'train',
  },
  predict: {
    group: 'predict',
  },
  cli: {
    group: 'cli',
  },
  reference: {
    group: 'reference',
  },
}

/*
 * The same single segment also serves the standalone pages that deliberately
 * carry no group prefix, because their URLs are permanent and short:
 * /docs/install, /docs/quickstart, /docs/licensing and the rest. Their markdown
 * lives in `content/docs/start/`, and a slug is only treated as one of these
 * when SECTIONS does not claim it first.
 */
const STANDALONE = 'start'

export function generateStaticParams() {
  return [
    ...Object.keys(SECTIONS).map((section) => ({ section })),
    ...getDocSlugs(STANDALONE).map((slug) => ({ section: slug })),
  ]
}

export async function generateMetadata({ params }) {
  const { locale, section } = await params
  const meta = SECTIONS[section]
  if (meta) {
    const t = await getTranslations({ locale, namespace: 'DocsSections' })
    return buildPageMetadata({
      title: t('metaTitle', { title: t(`${section}.title`) }),
      description: t(`${section}.description`),
      path: `/docs/${section}`,
      locale,
      englishOnly: false,
      ownImage: true,
    })
  }

  const doc = getDoc(STANDALONE, section, locale)
  if (!doc) return {}
  return {
    ...buildPageMetadata({
      title: doc.seo_title || doc.title,
      description: doc.description,
      path: `/docs/${section}`,
      locale,
      englishOnly: !doc.translated,
      ownImage: true,
    }),
    keywords: doc.keywords,
  }
}

export default async function SectionIndex({ params }) {
  const { locale, section } = await params
  setRequestLocale(locale)
  const t = await getTranslations({ locale, namespace: 'DocsSections' })
  const chrome = await getTranslations({ locale, namespace: 'DocsChrome' })
  const tiers = await getTranslations({ locale, namespace: 'Tiers' })
  const docsNav = localizeNav(locale)

  const meta = SECTIONS[section]
  if (!meta) return <StandalonePage locale={locale} slug={section} />

  const group = docsNav.groups.find((g) => g.id === meta.group)
  if (!group) notFound()

  // The section's own index entry would just link to this page.
  const items = group.items.filter((item) => item.slug !== `/docs/${section}`)
  const title = t(`${section}.title`)
  const breadcrumbs = [{ label: chrome('docsCrumb'), href: '/docs' }, { label: title }]

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbs.map((crumb, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: crumb.label,
      ...(crumb.href ? { item: `${SITE_URL}${crumb.href}` } : {}),
    })),
  }

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <DocsShell nav={docsNav} activePath={`/docs/${section}`} version={DOCS_VERSION} breadcrumbs={breadcrumbs} showActions={false}>
        <div className="max-w-3xl">
          <h1 className="text-[2.1rem] font-semibold tracking-tight text-surface-900 dark:text-white">
            {title}
          </h1>
          <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-surface-600 dark:text-surface-400">
            {t(`${section}.lead`)}
          </p>

          <dl className="mt-8 text-[14px]">
            {items.map((item) => {
              const tier = item.tier ? getTierMeta(item.tier) : null
              return (
                <div
                  key={item.slug}
                  className="flex flex-col gap-x-4 border-b border-surface-200/70 py-2 last:border-0 sm:flex-row dark:border-white/[0.07]"
                >
                  <dt className="shrink-0 sm:w-56">
                    {item.built ? (
                      <Link href={item.slug} className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                        {item.label}
                      </Link>
                    ) : (
                      <span className="text-surface-500 dark:text-surface-500">{item.label}</span>
                    )}
                  </dt>
                  <dd className="text-surface-500 dark:text-surface-500">
                    {tier ? tiers(`${item.tier}.label`) : ''}
                    {tier && !item.built ? '. ' : ''}
                    {!item.built ? chrome('notWrittenYet') : ''}
                  </dd>
                </div>
              )
            })}
          </dl>

          {group.more && (
            <p className="mt-4 text-[13px] text-surface-500 dark:text-surface-500">{group.more}.</p>
          )}
        </div>
      </DocsShell>
    </>
  )
}

/*
 * A prefix-free page such as /docs/install. Same shell, same header and the same
 * markdown pipeline as a sectioned page; only the breadcrumb is shorter, because
 * there is no group above it.
 */
async function StandalonePage({ locale, slug }) {
  const chrome = await getTranslations({ locale, namespace: 'DocsChrome' })
  const doc = getDoc(STANDALONE, slug, locale)
  if (!doc) notFound()

  const path = `/docs/${slug}`
  const url = localeUrl(path, doc.translated ? locale : routing.defaultLocale)
  const headings = extractHeadings(doc.content)
  const breadcrumbs = [{ label: chrome('docsCrumb'), href: '/docs' }, { label: doc.title }]

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
              {doc.verification || chrome('verified', { version: doc.last_verified })}
            </p>
          </footer>
        </article>
      </DocsShell>
    </>
  )
}
