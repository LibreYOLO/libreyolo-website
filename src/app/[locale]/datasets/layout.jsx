import { getTranslations } from 'next-intl/server'
import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('datasetsTitle'),
    description: t('datasetsDescription'),
    path: '/datasets',
    locale,
  })
}

export default function DatasetsLayout({ children }) {
  return children
}
