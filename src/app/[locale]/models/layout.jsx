import { getTranslations } from 'next-intl/server'
import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('modelsTitle'),
    description: t('modelsDescription'),
    path: '/models',
    locale,
  })
}

export default function ModelsLayout({ children }) {
  return children
}
