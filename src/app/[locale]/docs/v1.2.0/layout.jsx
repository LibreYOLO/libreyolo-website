import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  return buildPageMetadata({
    title: 'Docs v1.2.0 (previous release)',
    description: 'Documentation for LibreYOLO v1.2.0, the previous stable release: flagship YOLO9 and RF-DETR detection, training, validation, tiled inference, and ONNX / TensorRT / OpenVINO / NCNN export.',
    path: '/docs/v1.2.0',
    locale,
    englishOnly: true,
  })
}

export default function DocsV120Layout({ children }) {
  return children
}
