import { setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs } from '@/lib/docs'
import { buildPageMetadata } from '@/i18n/metadata'
import { SectionDocView } from '@/components/docs/DocViews'

// Export-format pages have no registry family: nothing on them is keyed to a
// model, so there is no ModelHeader, no hero and no generated block. The header
// rows come from the page's own frontmatter and the body is prose plus snippet
// groups.
const SECTION = 'export'

export function generateStaticParams() {
  return getDocSlugs(SECTION).map((slug) => ({ slug }))
}

export async function generateMetadata({ params }) {
  const { locale, slug } = await params
  const doc = getDoc(SECTION, slug, locale)
  if (!doc) return {}

  const path = `/docs/export/${slug}`
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

export default async function ExportDocPage({ params }) {
  const { locale, slug } = await params
  setRequestLocale(locale)
  return <SectionDocView locale={locale} section={SECTION} slug={slug} />
}
