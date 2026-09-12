import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  return buildPageMetadata({
    title: 'The LibreYOLO Sponsorship Program',
    description:
      'How LibreYOLO accepts money, what it does with it, and what sponsors receive. Version 1, September 2026.',
    path: '/sponsors',
    locale,
  })
}

export default function SponsorsLayout({ children }) {
  return children
}
