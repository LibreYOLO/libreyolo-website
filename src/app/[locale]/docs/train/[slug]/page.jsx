import { setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs, docAlternates } from '@/lib/docs'
import { buildPageMetadata } from '@/i18n/metadata'
import { SectionDocView } from '@/components/docs/DocViews'

// Workflow pages explain machinery shared across families, so nothing here is keyed to one model. Header rows come from frontmatter; runnable snippet groups come from the snippets map.
const SECTION = 'train'

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
      // translation that does not exist. Every translated version, English
      // included, lists the same hreflang set: English plus each twin.
      ...docAlternates(SECTION, slug, locale),
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
