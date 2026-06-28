import { getTranslations } from 'next-intl/server'
import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('scienceTitle'),
    description: t('scienceDescription'),
    path: '/science',
    locale,
  })
}

export default function ScienceLayout({ children }) {
  return children
}
