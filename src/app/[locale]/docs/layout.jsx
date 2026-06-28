import { getTranslations } from 'next-intl/server'
import { buildEnglishOnlyAlternates } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return {
    title: t('docsTitle'),
    description: t('docsDescription'),
    alternates: buildEnglishOnlyAlternates('/docs'),
  }
}

export default function DocsLayout({ children }) {
  return children
}
