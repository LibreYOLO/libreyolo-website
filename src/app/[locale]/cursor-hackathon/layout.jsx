import { buildEnglishOnlyAlternates, localeUrl, OG_IMAGE } from '@/i18n/metadata'

const description =
  'Everything you need for the LibreYOLO track of the Cursor Madrid Hackathon 3: intro, setup tutorial, and a working example.'

export function generateMetadata() {
  const title = 'LibreYOLO Track | Cursor Hackathon'
  return {
    title: { absolute: title },
    description,
    alternates: buildEnglishOnlyAlternates('/cursor-hackathon'),
    openGraph: {
      title,
      description,
      url: localeUrl('/cursor-hackathon', 'en'),
      siteName: 'LibreYOLO',
      locale: 'en_US',
      type: 'website',
      images: [OG_IMAGE],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [OG_IMAGE.url],
    },
  }
}

export default function CursorHackathonLayout({ children }) {
  return children
}
