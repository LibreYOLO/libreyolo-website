import { getDoc } from '@/lib/docs'
import { docsOgImage, OG_SIZE, OG_CONTENT_TYPE } from '@/lib/og-docs'

export const size = OG_SIZE
export const contentType = OG_CONTENT_TYPE
export const alt = 'LibreYOLO documentation'

export default async function Image({ params }) {
  const { locale, slug } = await params
  const doc = getDoc('tasks', slug, locale)
  return docsOgImage({
    title: doc?.title || slug,
    description: doc?.description || doc?.lead || '',
    section: 'tasks',
  })
}
