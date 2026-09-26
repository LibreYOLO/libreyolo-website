import { setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs } from '@/lib/docs'
import { buildPageMetadata } from '@/i18n/metadata'
import { SectionDocView } from '@/components/docs/DocViews'

// Task pages are capability hubs. They carry no single registry family (a task is served by many), so the header rows come from frontmatter and the model list is a generated block in the body.
const SECTION = 'tasks'

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
  return <SectionDocView locale={locale} section={SECTION} slug={slug} />
}
