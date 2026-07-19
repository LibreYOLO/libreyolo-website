import { buildPageMetadata } from '@/i18n/metadata'

const metadataByLocale = {
  en: {
    title: 'Docs v1.2.0 (previous release)',
    description: 'Documentation for LibreYOLO v1.2.0, the previous stable release: flagship YOLO9 and RF-DETR detection, training, validation, tiled inference, and ONNX / TensorRT / OpenVINO / NCNN export.',
  },
  zh: {
    title: 'v1.2.0 文档（上一版本）',
    description: 'LibreYOLO v1.2.0 上一稳定版本文档：旗舰 YOLO9 与 RF-DETR 检测、训练、验证、分块推理，以及 ONNX、TensorRT、OpenVINO 和 NCNN 导出。',
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  return buildPageMetadata({
    ...copy,
    path: '/docs/v1.2.0',
    locale,
  })
}

export default function DocsV120Layout({ children }) {
  return children
}
