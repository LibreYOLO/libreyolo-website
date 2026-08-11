import { buildPageMetadata } from '@/i18n/metadata'
import { getTranslations } from 'next-intl/server'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'LibreVLM' })
  return {
    ...buildPageMetadata({
      title: t('metaTitle'),
      description: t('metaDescription'),
      path: '/docs/librevlm',
      locale,
      englishOnly: false,
    }),
    keywords: t('keywords').split('|'),
  }
}

export default function LibreVLMDocsLayout({ children }) {
  return children
}
