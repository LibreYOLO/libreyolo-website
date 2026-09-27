import { setRequestLocale } from 'next-intl/server'

import { getDocsArchive } from '@/lib/docs'
import { archiveStaticParams, archivedDocsMetadata, ArchivedDocsView } from '@/components/docs/DocViews'

/*
 * Every page below /docs/v1.5.0: section indexes (/docs/v1.5.0/models),
 * standalone pages (/docs/v1.5.0/install) and sectioned pages
 * (/docs/v1.5.0/models/rf-detr). See ../page.jsx.
 */
const source = getDocsArchive('v1.5.0')

export function generateStaticParams() {
  return archiveStaticParams(source)
}

export async function generateMetadata({ params }) {
  const { locale, slug } = await params
  return archivedDocsMetadata({ locale, source, parts: slug })
}

export default async function DocsV150Page({ params }) {
  const { locale, slug } = await params
  setRequestLocale(locale)
  return <ArchivedDocsView locale={locale} source={source} parts={slug} />
}
