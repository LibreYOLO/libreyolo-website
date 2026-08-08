import { buildPageMetadata } from '@/i18n/metadata'
import FrozenVersionBanner from '@/components/docs/FrozenVersionBanner'

const metadataByLocale = {
  en: {
    title: 'Docs v1.3.1',
    description: 'Documentation for LibreYOLO v1.3.1, the current release: detection, instance and semantic segmentation, pose, classification, depth, image restoration, promptable segmentation with SAM, open-vocabulary detection, tracking, annotation with LibreLabel, distillation, training, validation, and ONNX / TensorRT / OpenVINO / NCNN / CoreML export.',
  },
  zh: {
    title: 'v1.3.1 文档',
    description: 'LibreYOLO v1.3.1 当前版本文档：检测、实例与语义分割、姿态、分类、深度、图像恢复、SAM 提示式分割、开放词表检测、跟踪、LibreLabel 标注、蒸馏、训练、验证，以及 ONNX、TensorRT、OpenVINO、NCNN 和 CoreML 导出。',
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  const meta = buildPageMetadata({
    ...copy,
    path: '/docs/v1.3.1',
    locale,
  })
  // Frozen version: canonicalise to the current docs tree. No noindex:
  // pairing noindex with a canonical is a documented way to deindex the
  // wrong page.
  return { ...meta, alternates: { canonical: 'https://www.libreyolo.com/docs' } }
}

export default function DocsV131Layout({ children }) {
  return (
    <>
      <FrozenVersionBanner version="v1.3.1" />
      {children}
    </>
  )
}
