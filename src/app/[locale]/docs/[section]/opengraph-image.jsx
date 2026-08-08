import { getDoc } from '@/lib/docs'
import { docsOgImage, OG_SIZE, OG_CONTENT_TYPE } from '@/lib/og-docs'

export const size = OG_SIZE
export const contentType = OG_CONTENT_TYPE
export const alt = 'LibreYOLO documentation'

/*
 * This segment serves two different things, so the card has to resolve which
 * one it is looking at: a section index (/docs/export) or a standalone page
 * whose slug sits directly under /docs (/docs/install). The page component
 * disambiguates the same way, by trying the standalone directory first and
 * treating a miss as a section.
 */
const SECTION_TITLE = {
  tasks: 'Tasks',
  models: 'Models',
  train: 'Training',
  predict: 'Prediction',
  export: 'Export and deploy',
  cli: 'CLI',
  reference: 'Reference',
}

export default async function Image({ params }) {
  const { locale, section } = await params
  const doc = getDoc('start', section, locale)

  if (doc) {
    return docsOgImage({
      title: doc.title,
      description: doc.description || doc.lead || '',
      section: 'start',
    })
  }

  return docsOgImage({
    title: SECTION_TITLE[section] || section,
    description: '',
    section,
  })
}
