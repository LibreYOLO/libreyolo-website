import { buildPageMetadata } from '@/i18n/metadata'

const metadataByLocale = {
  en: {
    title: 'Experimental Tasks | Classification, OBB, Pose, LoRA',
    description: 'Experimental LibreYOLO capabilities: image classification and oriented bounding boxes (OBB) for YOLO9 and RF-DETR, keypoint and pose estimation, and LoRA/DoRA fine-tuning for RF-DETR.',
    keywords: ['LibreYOLO experimental', 'image classification', 'oriented bounding boxes', 'OBB', 'pose estimation', 'keypoints', 'LoRA', 'DoRA', 'RF-DETR fine-tuning', 'YOLO9'],
  },
  zh: {
    title: '实验性任务 | 分类、OBB、姿态与 LoRA',
    description: 'LibreYOLO 实验性功能：YOLO9 与 RF-DETR 的图像分类和旋转框（OBB）、关键点与姿态估计，以及 RF-DETR 的 LoRA/DoRA 微调。',
    keywords: ['LibreYOLO 实验性功能', '图像分类', '旋转框', 'OBB', '姿态估计', '关键点', 'LoRA', 'DoRA', 'RF-DETR 微调', 'YOLO9'],
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  return {
    ...buildPageMetadata({
      title: copy.title,
      description: copy.description,
      path: '/docs/experimental',
      locale,
    }),
    keywords: copy.keywords,
  }
}

export default function ExperimentalDocsLayout({ children }) {
  return children
}
