import { getTranslations, setRequestLocale } from 'next-intl/server'

import { buildPageMetadata } from '@/i18n/metadata'
import { DocsLandingView } from '@/components/docs/DocViews'

/*
 * Docs landing. The page body lives in DocViews, shared with the frozen
 * release trees; see the comment there.
 */

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'DocsLanding' })
  return buildPageMetadata({
    title: t('metaTitle'),
    description: t('metaDescription'),
    path: '/docs',
    locale,
    englishOnly: false,
  })
}

export default async function DocsLanding({ params }) {
  const { locale } = await params
  setRequestLocale(locale)
  return <DocsLandingView locale={locale} />
}
