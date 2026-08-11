import { buildPageMetadata } from '@/i18n/metadata'
import { getTranslations } from 'next-intl/server'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Experimental' })
  return {
    ...buildPageMetadata({
      title: t('metaTitle'),
      description: t('metaDescription'),
      path: '/docs/experimental',
      locale,
      englishOnly: false,
    }),
    keywords: t('keywords').split('|'),
  }
}

export default function ExperimentalDocsLayout({ children }) {
  return children
}
