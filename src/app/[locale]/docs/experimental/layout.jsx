import { buildEnglishOnlyAlternates } from '@/i18n/metadata'

export const metadata = {
  title: 'Experimental Tasks | Classification, OBB, Pose, LoRA',
  description:
    'Experimental LibreYOLO capabilities: image classification and oriented bounding boxes (OBB) for YOLO9 and RF-DETR, keypoint and pose estimation, and LoRA/DoRA fine-tuning for RF-DETR.',
  keywords: [
    'LibreYOLO experimental', 'image classification', 'oriented bounding boxes', 'OBB',
    'pose estimation', 'keypoints', 'LoRA', 'DoRA', 'RF-DETR fine-tuning', 'YOLO9',
  ],
  alternates: buildEnglishOnlyAlternates('/docs/experimental'),
}

export default function ExperimentalDocsLayout({ children }) {
  return children
}
