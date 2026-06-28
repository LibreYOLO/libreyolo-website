import { getTranslations } from 'next-intl/server'
import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('docsTitle'),
    description: t('docsDescription'),
    path: '/docs',
    locale,
    englishOnly: true,
  })
}

export default function DocsLayout({ children }) {
  return children
}
