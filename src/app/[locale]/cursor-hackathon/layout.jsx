import { buildAlternates } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  return {
    title: { absolute: 'LibreYOLO Track | Cursor Hackathon' },
    description: 'Everything you need for the LibreYOLO track of the Cursor Madrid Hackathon 3: intro, setup tutorial, and a working example.',
    alternates: buildAlternates('/cursor-hackathon', locale),
  }
}

export default function CursorHackathonLayout({ children }) {
  return children
}
