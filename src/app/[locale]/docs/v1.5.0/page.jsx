import { setRequestLocale } from 'next-intl/server'

import { getDocsArchive } from '@/lib/docs'
import { archivedDocsMetadata, ArchivedDocsView } from '@/components/docs/DocViews'

/*
 * Frozen docs for LibreYOLO 1.5.0: the whole v2 tree as it stood at release,
 * rendered by the same views as /docs from the snapshot in
 * content/archive/docs/v1.5.0 and src/data/docs/archive/v1.5.0. Every page
 * canonicalises to its current counterpart and none is in the sitemap.
 */
const source = getDocsArchive('v1.5.0')

export async function generateMetadata({ params }) {
  const { locale } = await params
  return archivedDocsMetadata({ locale, source, parts: [] })
}

export default async function DocsV150Landing({ params }) {
  const { locale } = await params
  setRequestLocale(locale)
  return <ArchivedDocsView locale={locale} source={source} parts={[]} />
}
