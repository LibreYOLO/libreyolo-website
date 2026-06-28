import { buildEnglishOnlyAlternates } from '@/i18n/metadata'

export function generateMetadata() {
  return {
    title: { absolute: 'LibreYOLO Track | Cursor Hackathon' },
    description: 'Everything you need for the LibreYOLO track of the Cursor Madrid Hackathon 3: intro, setup tutorial, and a working example.',
    alternates: buildEnglishOnlyAlternates('/cursor-hackathon'),
  }
}

export default function CursorHackathonLayout({ children }) {
  return children
}
