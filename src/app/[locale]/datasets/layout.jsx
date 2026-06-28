import { getTranslations } from 'next-intl/server'
import { buildAlternates } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return {
    title: t('datasetsTitle'),
    description: t('datasetsDescription'),
    alternates: buildAlternates('/datasets', locale),
  }
}

export default function DatasetsLayout({ children }) {
  return children
}
