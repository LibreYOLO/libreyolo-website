import { buildPageMetadata } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  return buildPageMetadata({
    title: 'Docs v1.3.1',
    description: 'Documentation for LibreYOLO v1.3.1, the current release: detection, instance and semantic segmentation, pose, classification, depth, image restoration, promptable segmentation with SAM, open-vocabulary detection, tracking, annotation with LibreLabel, distillation, training, validation, and ONNX / TensorRT / OpenVINO / NCNN / CoreML export.',
    path: '/docs/v1.3.1',
    locale,
    englishOnly: true,
  })
}

export default function DocsV131Layout({ children }) {
  return children
}
