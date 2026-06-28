import { buildEnglishOnlyAlternates } from '@/i18n/metadata'

export const metadata = {
  title: 'Docs v1.1.0',
  description: 'Archived documentation for LibreYOLO v1.1.0: installation, quickstart, training, validation, export, and inference backends.',
  alternates: buildEnglishOnlyAlternates('/docs/v1.1.0'),
}

export default function DocsV110Layout({ children }) {
  return children
}
