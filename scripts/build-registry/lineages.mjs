/*
 * The full lineage table: one docs page per upstream lineage.
 *
 * Families are merged onto one page only where they are versions of the same
 * upstream work (YOLOv9 and its E2E and P2 variants; RT-DETR v1/v2/v4; DEIM and
 * DEIMv2). Everything else gets its own page, because a reader searching a
 * model name expects to land on a page about that model.
 *
 * Slugs are the term someone types into a search box: recognizable upstream
 * names, lowercase and hyphenated, never the internal registry key where the
 * two differ (`ec` is EdgeCrafter, `ppocr` is PP-OCRv5).
 */
export const LINEAGES = [
  // g0, flagships
  { slug: 'yolov9', display: 'YOLOv9', keys: ['yolo9', 'yolo9_e2e', 'yolo9_p2'] },
  { slug: 'rf-detr', display: 'RF-DETR', keys: ['rfdetr'] },

  // g1, core trainable detectors
  { slug: 'edgecrafter', display: 'EdgeCrafter', keys: ['ec'] },
  { slug: 'rt-detr', display: 'RT-DETR', keys: ['rtdetr', 'rtdetrv2', 'rtdetrv4'] },
  { slug: 'd-fine', display: 'D-FINE', keys: ['dfine'] },
  { slug: 'deim', display: 'DEIM', keys: ['deim', 'deimv2'] },
  { slug: 'yolo-nas', display: 'YOLO-NAS', keys: ['yolonas'] },

  // g2, supporting trainables
  { slug: 'yolox', display: 'YOLOX', keys: ['yolox'] },
  { slug: 'yolov7', display: 'YOLOv7', keys: ['yolo7'] },
  { slug: 'rtmdet', display: 'RTMDet', keys: ['rtmdet'] },
  { slug: 'picodet', display: 'PicoDet', keys: ['picodet'] },
  { slug: 'fomo', display: 'FOMO', keys: ['fomo'] },
  { slug: 'dome-detr', display: 'Dome-DETR', keys: ['domedetr'] },
  { slug: 'segformer', display: 'SegFormer', keys: ['segformer'] },
  { slug: 'lingbot-vision', display: 'LingBot-Vision', keys: ['lingbotvision'] },
  { slug: 'dinov2', display: 'DINOv2', keys: ['dinov2'] },
  { slug: 'nafnet', display: 'NAFNet', keys: ['nafnet'] },
  { slug: 'resnet', display: 'ResNet', keys: ['resnet'] },
  { slug: 'convnext', display: 'ConvNeXt', keys: ['convnext'] },
  { slug: 'mobilenetv4', display: 'MobileNetV4', keys: ['mobilenetv4'] },
  { slug: 'efficientnetv2', display: 'EfficientNetV2', keys: ['efficientnetv2'] },

  // g3, inference-only specialists
  { slug: 'lw-detr', display: 'LW-DETR', keys: ['lwdetr'] },
  { slug: 'detr', display: 'DETR', keys: ['detr'] },
  { slug: 'deformable-detr', display: 'Deformable DETR', keys: ['deformable_detr'] },
  { slug: 'dino-detr', display: 'DINO-DETR', keys: ['dinodetr'] },
  { slug: 'mask-rcnn', display: 'Mask R-CNN', keys: ['mask_rcnn'] },
  { slug: 'faster-rcnn', display: 'Faster R-CNN', keys: ['faster_rcnn'] },
  { slug: 'fcos', display: 'FCOS', keys: ['fcos'] },
  { slug: 'retinanet', display: 'RetinaNet', keys: ['retinanet'] },
  { slug: 'ssd', display: 'SSD', keys: ['ssd'] },
  { slug: 'centernet', display: 'CenterNet', keys: ['centernet'] },
  { slug: 'efficientdet', display: 'EfficientDet', keys: ['efficientdet'] },
  { slug: 'vit', display: 'ViT', keys: ['vit'] },
  { slug: 'swin', display: 'Swin Transformer', keys: ['swin'] },
  { slug: 'vgg', display: 'VGG', keys: ['vgg'] },
  { slug: 'alexnet', display: 'AlexNet', keys: ['alexnet'] },
  { slug: 'fcn', display: 'FCN', keys: ['fcn'] },
  { slug: 'deeplabv3', display: 'DeepLabv3', keys: ['deeplabv3'] },
  { slug: 'pidnet', display: 'PIDNet', keys: ['pidnet'] },
  { slug: 'eomt', display: 'EoMT', keys: ['eomt'] },
  { slug: 'hrnet', display: 'HRNet', keys: ['hrnet'] },
  { slug: 'l2cs', display: 'L2CS-Net', keys: ['l2cs'] },
  { slug: 'depth-anything-v2', display: 'Depth Anything V2', keys: ['depth_anything'] },
  { slug: 'depth-anything-3', display: 'Depth Anything 3', keys: ['depth_anything3'] },
  { slug: 'zipdepth', display: 'ZipDepth', keys: ['zipdepth'] },
  { slug: 'midas', display: 'MiDaS', keys: ['midas'] },
  { slug: 'moge-2', display: 'MoGe-2', keys: ['moge2'] },
  { slug: 'teed', display: 'TEED', keys: ['teed'] },
  { slug: 'dexined', display: 'DexiNed', keys: ['dexined'] },
  { slug: 'swinir', display: 'SwinIR', keys: ['swinir'] },
  { slug: 'real-esrgan', display: 'Real-ESRGAN', keys: ['realesrgan'] },
  { slug: 'birefnet', display: 'BiRefNet', keys: ['birefnet'] },
  { slug: 'feynobg', display: 'FeyNobg', keys: ['feynobg'] },
  { slug: 'pp-ocrv5', display: 'PP-OCRv5', keys: ['ppocr'] },
  { slug: 'librefacerec', display: 'LibreFaceRec', keys: ['facerec'] },
  { slug: 'sam-3d-body', display: 'SAM 3D Body', keys: ['sam3dbody'] },

  // g4, museum
  { slug: 'yolov1', display: 'YOLOv1', keys: ['yolo1'] },
  { slug: 'yolov2', display: 'YOLOv2', keys: ['yolo2'] },
  { slug: 'yolov3', display: 'YOLOv3', keys: ['yolo3'] },
  { slug: 'yolov4', display: 'YOLOv4', keys: ['yolo4'] },
  { slug: 'deit', display: 'DeiT', keys: ['deit'] },

  // s, sibling tiers
  { slug: 'sam', display: 'SAM', keys: ['sam'] },
  { slug: 'sam-2', display: 'SAM 2', keys: ['sam2'] },
  { slug: 'sam-3', display: 'SAM 3', keys: ['sam3'] },
  { slug: 'mobilesam', display: 'MobileSAM', keys: ['mobilesam'] },
  { slug: 'edgetam', display: 'EdgeTAM', keys: ['edgetam'] },
  { slug: 'picosam3', display: 'PicoSAM3', keys: ['picosam3'] },
  { slug: 'grounding-dino', display: 'Grounding DINO', keys: ['grounding_dino'] },
  { slug: 'owlv2', display: 'OWLv2', keys: ['owlv2'] },
  { slug: 'omdet-turbo', display: 'OMDet-Turbo', keys: ['omdet_turbo'] },
  { slug: 'ov-deim', display: 'OV-DEIM', keys: ['ov_deim'] },
  { slug: 'florence-2', display: 'Florence-2', keys: ['florence2'] },
  { slug: 'kosmos-2', display: 'Kosmos-2', keys: ['kosmos2'] },
  { slug: 'qwen3-vl', display: 'Qwen3-VL', keys: ['qwen3vl'] },
  { slug: 'smolvlm2', display: 'SmolVLM2', keys: ['smolvlm2'] },
  { slug: 'internvl3', display: 'InternVL3', keys: ['internvl3'] },
  { slug: 'lfm2-vl', display: 'LFM2-VL', keys: ['lfm2vl'] },
  { slug: 'locate-anything', display: 'LocateAnything', keys: ['locateanything'] },
  { slug: 'clip', display: 'CLIP', keys: ['clip'] },
  { slug: 'siglip2', display: 'SigLIP2', keys: ['siglip2'] },
  { slug: 'sensenova-vision', display: 'SenseNova-Vision', keys: ['sensenovavision'] },
  { slug: 'libremodus', display: 'LibreMODUS', keys: ['libremodus'] },
]
