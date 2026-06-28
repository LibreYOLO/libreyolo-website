import { getTranslations } from 'next-intl/server'
import { buildAlternates } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return {
    title: t('modelsTitle'),
    description: t('modelsDescription'),
    alternates: buildAlternates('/models', locale),
  }
}

export default function ModelsLayout({ children }) {
  return children
}
