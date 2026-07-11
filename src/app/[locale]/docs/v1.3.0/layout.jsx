import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  return buildPageMetadata({
    title: 'Docs v1.3.0 (previous release)',
    description: 'Documentation for LibreYOLO v1.3.0, the previous release: model families for classification, depth, and point localization, RF-DETR pose and oriented boxes, new CLI tools, training loggers, and export updates. For the current release see the v1.3.1 docs.',
    path: '/docs/v1.3.0',
    locale,
    englishOnly: true,
  })
}

export default function DocsV130Layout({ children }) {
  return children
}
