import { getTranslations } from 'next-intl/server'
import { buildAlternates } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return {
    title: t('scienceTitle'),
    description: t('scienceDescription'),
    alternates: buildAlternates('/science', locale),
  }
}

export default function ScienceLayout({ children }) {
  return children
}
