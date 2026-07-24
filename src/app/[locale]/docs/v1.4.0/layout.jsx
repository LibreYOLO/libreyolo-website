import { buildPageMetadata } from '@/i18n/metadata'

const metadataByLocale = {
  en: {
    title: 'Docs v1.4.0',
    description: 'Documentation for LibreYOLO v1.4.0, the current release: detection, instance, semantic and panoptic segmentation, pose, classification, depth, super-resolution and restoration, background removal, OCR, promptable segmentation with SAM, open-vocabulary detection, tracking, training augmentation, quantization, distillation, training, validation, and ONNX / TensorRT / OpenVINO / NCNN / CoreML / TFLite export.',
  },
  zh: {
    title: 'v1.4.0 文档',
    description: 'LibreYOLO v1.4.0 当前版本文档：检测、实例分割、语义分割、全景分割、姿态、分类、深度、超分辨率与图像修复、背景移除、OCR、SAM 提示式分割、开放词表检测、跟踪、训练数据增强、量化、蒸馏、训练、验证，以及 ONNX、TensorRT、OpenVINO、NCNN、CoreML 和 TFLite 导出。',
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  return buildPageMetadata({
    ...copy,
    path: '/docs/v1.4.0',
    locale,
  })
}

export default function DocsV140Layout({ children }) {
  return children
}
