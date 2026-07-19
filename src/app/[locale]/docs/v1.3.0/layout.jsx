import { buildPageMetadata } from '@/i18n/metadata'

const metadataByLocale = {
  en: {
    title: 'Docs v1.3.0 (previous release)',
    description: 'Documentation for LibreYOLO v1.3.0, the previous release: model families for classification, depth, and point localization, RF-DETR pose and oriented boxes, new CLI tools, training loggers, and export updates. For the current release see the v1.3.1 docs.',
  },
  zh: {
    title: 'v1.3.0 文档（上一版本）',
    description: 'LibreYOLO v1.3.0 上一版本文档：分类、深度和点定位模型系列、RF-DETR 姿态与旋转框、新增 CLI 工具、训练日志记录器以及导出更新。当前版本请参阅 v1.3.1 文档。',
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  return buildPageMetadata({
    ...copy,
    path: '/docs/v1.3.0',
    locale,
  })
}

export default function DocsV130Layout({ children }) {
  return children
}
