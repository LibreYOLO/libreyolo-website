// Single source of truth for the Model Zoo showcase: every task shown on
// /models with its model-family list. The models page renders these and the
// models layout derives its ItemList JSON-LD from them, so the structured
// data can never drift from what the page visibly claims.

// Same parkour clip, three tasks, in motion. Each carries its model list, shown
// under the video (so these tasks don't need their own section below).
export const VIDEOS = [
  { src: '/showcase/parkour-detection.mp4', poster: '/showcase/parkour-detection-poster.jpg', label: 'Detection', models: ['YOLO9', 'RF-DETR', 'YOLOX', 'YOLO9-E2E', 'YOLO9-P2', 'YOLO-NAS', 'D-FINE', 'DEIM', 'DEIMv2', 'RT-DETR', 'RT-DETRv2', 'RT-DETRv4', 'PicoDet', 'RTMDet', 'EdgeCrafter'] },
  { src: '/showcase/parkour-segmentation.mp4', poster: '/showcase/parkour-segmentation-poster.jpg', label: 'Segmentation', models: ['RF-DETR', 'EdgeCrafter', 'SAM', 'MobileSAM', 'SAM2'] },
  { src: '/showcase/parkour-pose.mp4', poster: '/showcase/parkour-pose-poster.jpg', label: 'Keypoints', models: ['RF-DETR', 'EdgeCrafter', 'YOLO-NAS'] },
]

// The remaining tasks, each with an example image + model list.
export const TASKS = [
  { title: 'Classification', image: '/showcase/task-classification.jpg', models: ['ConvNeXt', 'EfficientNetV2', 'MobileNetV4', 'DINOv2', 'ResNet', 'CLIP'] },
  { title: 'Oriented boxes', image: '/showcase/task-obb.jpg', models: ['RF-DETR'] },
  { title: 'Point & counting', image: '/showcase/task-point.jpg', models: ['FOMO'] },
  { title: 'Gaze', image: '/showcase/task-gaze.gif', models: ['L2CS'] },
  { title: 'Vision-language', image: '/showcase/task-vlm.jpg', models: ['Florence-2', 'InternVL3', 'Kosmos-2', 'LFM2-VL', 'Qwen3-VL', 'SmolVLM2'] },
  { title: 'Depth', image: '/showcase/depth-reveal.gif', models: ['Depth Anything V2'] },
]

// [{ task, models }] across both showcase groups, in display order.
export function showcaseTaskGroups() {
  return [
    ...VIDEOS.map(({ label, models }) => ({ task: label, models })),
    ...TASKS.map(({ title, models }) => ({ task: title, models })),
  ]
}
