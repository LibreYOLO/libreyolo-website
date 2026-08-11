import { getTranslations } from 'next-intl/server'
import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'CursorHackathon' })
  return buildPageMetadata({
    title: t('metaTitle'),
    description: t('metaDescription'),
    path: '/cursor-hackathon',
    locale,
    englishOnly: false,
  })
}

export default function CursorHackathonLayout({ children }) {
  return children
}
