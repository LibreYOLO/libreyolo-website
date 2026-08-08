---
title: Changelog
seo_title: "LibreYOLO changelog: recent releases"
description: "What landed in LibreYOLO since 1.4.0 and what 1.4.0 contained: new model families, CUDA graph coverage, faster COCO metrics and training fixes."
lead: "A summary of the repository changelog: the work recorded since 1.4.0, and what 1.4.0 itself shipped."
keywords: [libreyolo changelog, libreyolo release notes, libreyolo 1.4.0, libreyolo new models]
last_verified: "1.5.0"
---

## Since 1.4.0

Summarized from
[`CHANGELOG.md`](https://github.com/LibreYOLO/libreyolo/blob/dev/CHANGELOG.md)
on the `dev` branch, which carries the full entries.

### New model families

Dome-DETR (`domedetr`) is the trainable addition: a tiny-object detector for
aerial, drone and remote-sensing imagery, ported from the Apache-2.0 Dome-DETR
release. It is D-FINE plus a density head (DeFE), encoder attention restricted
to occupied windows (MWAS), and a query count set by local density rather than
a fixed 300 (PAQI). Sizes s, m and l at 800x800, with a maximum absolute
difference of 0.0 against upstream on all six published checkpoints. Its
advantage narrows as objects grow, so it sits beside D-FINE rather than
replacing it. There is no COCO checkpoint upstream, only AI-TOD-V2 with 9
classes and VisDrone with 12, so every canonical filename carries a dataset
suffix. Export raises instead of emitting a graph, because PAQI's per-image
query count makes a traced graph valid only for the image it was traced on. The
code is Apache-2.0, but the weights are not rehosted: the upstream model card
states no license in its metadata while its prose claims Apache-2.0 and
restricts use to academic research.

The rest are inference-only ports of earlier published work, each checked
against its pinned upstream source:

- Detection: DETR, Deformable DETR, DINO-DETR, LW-DETR, Faster R-CNN,
  Mask R-CNN, FCOS, RetinaNet, SSD300, CenterNet and EfficientDet.
- Classification: AlexNet, VGG, ViT, DeiT and Swin.
- Semantic segmentation: FCN and DeepLabv3.
- Depth: MiDaS. Pose: HRNet, top-down on COCO-17 person crops.

TEED and DexiNed arrived with the new `edge` task, as native MIT-licensed
architectures with local checkpoint converters. Their upstream BIPED-trained
checkpoints are not bundled, mirrored or auto-downloaded, because that dataset's
terms are non-commercial.

LibreMODUS adds 14B-A7B analysis-only inference for depth, normals, edges,
COCO detection and phrase grounding, with image-conditioned `any2any()`
chaining. LibreFeyNobg adds a matte family at 1024 px, code and weights
Apache-2.0.

### New tasks and outputs

The `edge` and `normal` dense-prediction task contracts landed: original-canvas
result payloads and visualization, dataset schemas, validators with edge
ODS/OIS and normal angular metrics, and public API aliases.

RT-DETRv2 gained oriented-object detection inference for the official DOTA 1.0
checkpoints in sizes n, s, m, l and x, with aspect-preserving preprocessing,
native `Results.obb` output, validation, and validated ONNX and TorchScript
export.

### Speed

CUDA graph capture of the training step went from 2 families to 24, across
detect, classify, semantic, point and restore. Measured on an RTX 5070 Ti under
AMP: 3.63x on FOMO, 2.74x on MobileNetV4, 1.99x on YOLO9-t, and 1.04x to 1.26x
for the rest, with the win tracking how much of a step is network rather than
loss. End to end on a 20-epoch YOLO9-t fine-tune of 406 images, dataloader and
validation included, 428 s became 368 s with identical mAP50-95 and per-epoch
losses.

Capture at prediction time went from 8 families to 39, spanning detect,
segment, pose, point, classify, semantic, depth, restore, matte and OCR. Every
enabled family is verified to replay bit-identically against two probe inputs.
A family that cannot be captured whole is split at a verified seam, and the
remainder runs eagerly with identical numbers.

`pip install libreyolo[hub-kernels]` opts every Deformable-DETR-lineage family
into a compiled Apache-2.0 CUDA kernel for multi-scale deformable attention,
pinned to an audited revision. It applies to eager CUDA fp32 only; exports keep
the portable path.

Fused scaled dot product attention now runs across the transformer families
using stock torch, with no optional dependency. Measured on an RTX 5070 Ti
under fp16 autocast, roughly 2x on Swin window attention and 3.8x on OWLv2
vision attention. Families held to a byte-exact parity bar keep manual
attention by default and opt in explicitly.

COCO metrics moved to faster-coco-eval by default: 15.6x faster overall and 56x
on detection-dense datasets, decided on measured parity across all 100 RF100-VL
test splits, where 1381 of 1400 metric values were bit-identical to
pycocotools, the maximum deviation was 2.22e-16, and headline deltas were
exactly 0. `--no-faster-coco-eval` opts out, and pycocotools stays the automatic
fallback when the package is missing.

### Training and validation

`val_loss=True` reached every trainable family, across detect, classify,
semantic and restore rather than detection alone. It moved from `YOLO9Config`
and `RFDETRConfig` to `TrainConfig`, so a family that has not implemented it
now raises a clear error instead of ignoring the flag. Denoising terms are
never included, because validation forwards without ground truth.

Comet, ClearML, Neptune and DVCLive joined the built-in training loggers, with
the same canonical metrics and failure-isolation contract as the existing
TensorBoard, MLflow and Weights & Biases integrations.

The kernel registry moved to `libreyolo/kernels/`, organized by purpose.
`LIBREYOLO_KERNELS` replaces `LIBREYOLO_QUANT_KERNELS`, which is still honored,
and `libreyolo.quant.kernels` remains a working alias.

### Fixes

`train(..., cuda_graph=True)` had been a silent no-op for D-FINE, DEIM, DEIMv2,
RT-DETRv4 and EC, whose trainers called the eager forward instead of the routed
one, so capture never engaged. The end-to-end suite now asserts that capture
actually engages per family.

YOLOX applies its BatchNorm `eps=1e-3` and `momentum=0.03` at construction
instead of as a fixup afterwards, so the values survive the class-count rebuild
that `train()` performs when a dataset's class count differs from the
checkpoint. A fine-tune previously trained and reported in-training validation
at torch's default `eps=1e-5` but was reloaded for inference at `1e-3`. On
RF100-VL `ball`, the same nano checkpoint scores 0.566 mAP50-95 evaluated at
its trained eps and 0.151 after a stock reload. Checkpoints trained before this
fix carry the old semantics and need the override to report faithful numbers.

Non-strict checkpoint loads now warn with the counts and first names of missing
and unexpected state-dict keys. Shape mismatches always raised, but name
mismatches let a partially matching checkpoint load and then predict with
freshly initialized tensors, leaving no trace.

CUDA graph capture no longer races with DataLoader pin-memory threads, which
had killed runs with a capture error from the pin-memory thread. D-FINE
training now applies upstream's per-size multi-scale recipe instead of a
hardcoded `base_size_repeat=3`.

## 1.4.0, released 2026-07-24

The release notes summarize it as 15 new model families, 3 new tasks, a
quantization stack, two new trackers, and a multi-GPU training correctness
overhaul.

The new families were SegFormer, SwinIR, Real-ESRGAN, BiRefNet, ZipDepth,
Depth Anything 3, PP-OCRv5, SigLIP 2, YOLOv1, SAM 3, EdgeTAM, PicoSAM3,
OMDet-Turbo, OV-DEIM and SenseNova-Vision. The new tasks were `panoptic`,
`matte` and `ocr`, each with its result type and validator, and EoMT gained
instance and panoptic segmentation.

Quantization arrived as the `libreyolo quantize` command and `model.quantize()`,
with fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4 and mxfp4 recipes, and
quantization-aware training through `train()` on a quantized checkpoint, for
yolo9 and rfdetr. Tracking gained BoT-SORT and Deep OC-SORT with an OSNet-AIN
re-identification embedder.

The multi-GPU work was about correctness. DDP now shards correctly for DEIM,
D-FINE and YOLO-NAS-pose, where every rank had been training the full dataset
at the full batch, and loss normalizers are globally all-reduced to match
single-GPU gradients. SyncBatchNorm defaults on for the CNN detectors, and a
non-divisible global batch or a custom loader that does not shard now raises at
setup instead of running wrong.

Several training fixes changed results outright: RTMDet fine-tune collapse from
missing head init took an nc=1 rebuild from 0.26 to 0.709 mAP50-95, YOLOv7
training color-space and fp16 overflow bugs took a fine-tune from 0.0 to 0.92
mAP50, and PicoDet's and DEIM's learning-rate defaults were lowered because the
old ones destroyed pretrained weights. Segmentation training no longer exhausts
RAM on COCO-scale datasets with multiple workers.

One compatibility note from that release: checkpoints written with the new task
strings or with finalized quantization state cannot be loaded by 1.3.1.

## Earlier releases

Releases before 1.4.0 are documented in the
[GitHub releases](https://github.com/LibreYOLO/libreyolo/releases) only, not in
`CHANGELOG.md`. Their documentation is still online, listed on
[versions](/docs/versions).
