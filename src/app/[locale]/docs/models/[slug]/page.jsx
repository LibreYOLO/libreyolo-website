import { setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs } from '@/lib/docs'
import { buildPageMetadata } from '@/i18n/metadata'
import { SectionDocView } from '@/components/docs/DocViews'

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
      ownImage: true,
    }),
    keywords: doc.keywords,
  }
}

export default async function ModelDocPage({ params }) {
  const { locale, slug } = await params
  setRequestLocale(locale)
  return <SectionDocView locale={locale} section={SECTION} slug={slug} />
}
