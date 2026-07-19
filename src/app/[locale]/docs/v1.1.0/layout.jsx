import { buildPageMetadata } from '@/i18n/metadata'

const metadataByLocale = {
  en: {
    title: 'Docs v1.1.0',
    description: 'Archived documentation for LibreYOLO v1.1.0: installation, quickstart, training, validation, export, and inference backends.',
  },
  zh: {
    title: 'v1.1.0 文档',
    description: 'LibreYOLO v1.1.0 归档文档：安装、快速开始、训练、验证、导出以及推理后端。',
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  return buildPageMetadata({
    ...copy,
    path: '/docs/v1.1.0',
    locale,
  })
}

export default function DocsV110Layout({ children }) {
  return children
}
