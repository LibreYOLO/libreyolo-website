import { getTranslations } from 'next-intl/server'
import { buildAlternates } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return {
    title: t('commercialTitle'),
    description: t('commercialDescription'),
    alternates: buildAlternates('/commercial', locale),
  }
}

export default function CommercialLayout({ children }) {
  return children
}
