---
title: Changelog
seo_title: "LibreYOLO changelog: recent releases"
description: "What landed in LibreYOLO 1.5.0 and what 1.4.0 contained: 28 new model families, four new tasks, five export formats, torch-free ONNX inference and faster COCO metrics."
lead: "A summary of the repository changelog: what 1.5.0 shipped, and what 1.4.0 shipped before it."
keywords: [libreyolo changelog, libreyolo release notes, libreyolo 1.5.0, libreyolo 1.4.0, libreyolo new models]
last_verified: "1.5.0"
---

## 1.5.0, released 2026-08-09

The largest release so far: 28 new model families, four new tasks, five new
export formats, two new serving backends, and an `import libreyolo` that no
longer pulls PyTorch. 600 commits and 88 merged pull requests.

Summarized from
[`CHANGELOG.md`](https://github.com/LibreYOLO/libreyolo/blob/release/CHANGELOG.md),
which carries the full entries. Two defaults move numbers and four arguments
changed shape: [upgrading to 1.5.0](/docs/upgrade) is the short list of what
that asks of you.

### Breaking changes

No public model class or function was removed, and `__all__` grew from 101
names to 142. Four things need a code edit:

- `allow_experimental=True` is gone from every `.train()` gate. Delete the
  argument; a call that still passes it raises `TypeError`.
- The export support tier `"experimental"` is gone. `Tier` is now
  `validated`, `available` or `blocked`.
- `pretrained=False` combined with `resume` raises `ValueError` instead of
  proceeding incoherently.
- CLI `--imgsz` widened from int to str so it can carry `480x640`. Typing
  `--imgsz 640` in a shell and calling `model.predict(imgsz=640)` are both
  unaffected; only direct Python calls into the CLI command functions need a
  string.

Three changes move metrics at default settings: the COCO backend, YOLOX
BatchNorm eps on checkpoints trained before this release, and D-FINE's
per-size multi-scale recipe. All of it, with before and after code, is on
[upgrading to 1.5.0](/docs/upgrade).

### New tasks

The task vocabulary went from 13 entries to 17, with nothing removed. Each new
task ships an original-canvas result payload, visualization, dataset schema and
filename suffix.

[`edge`](/docs/tasks/edge-detection) and
[`normal`](/docs/tasks/surface-normals) are dense-prediction contracts, with
validators for edge ODS/OIS and normal angular error.

[`embed`](/docs/tasks/face-recognition) produces L2-normalized image and region
embeddings. It ships `LibreFaceEmbedder`, a `Gallery` and `FaceGallery` API,
and 1:N identification from `libreyolo predict` through `--gallery` and
`--gallery-threshold`. The task is deliberately general: identity vectors are
one use, region embeddings and re-identification are others.

[`mesh`](/docs/tasks/body-mesh) is body mesh recovery. `LibreSAM3DBody` is an
optional model gated on the `sam_3d_body` dependency, so it is not exported
from the top-level package.

### New model families

28 new user-facing classes, exported under 29 names. All are inference-only
unless stated.

Dome-DETR (`domedetr`) is the trainable addition: a tiny-object detector for
aerial, drone and remote-sensing imagery, ported from the Dome-DETR release. It
is D-FINE plus a density head (DeFE), encoder attention restricted to occupied
windows (MWAS), and a query count set by local density rather than a fixed 300
(PAQI). Sizes s, m and l at 800x800, with a maximum absolute difference of 0.0
against upstream on all six published checkpoints. Training is wired against
upstream's full objective, though the published 160-epoch schedule has not been
reproduced, so the paper's AP numbers are unverified. Its advantage narrows as
objects grow, so it sits beside D-FINE rather than replacing it. There is no
COCO checkpoint upstream, only AI-TOD-V2 with 9 classes and VisDrone with 12,
so every canonical filename carries a dataset suffix. Export raises instead of
emitting a graph, because PAQI's per-image query count makes a traced graph
valid only for the image it was traced on. Weights are not rehosted: the
upstream card claims Apache-2.0 while also restricting use to academic
research.

[LingBot-Vision](/docs/models/lingbot-vision) is the other trainable arrival, a
ViT semantic-segmentation family at 512 px pairing boundary-centric
self-supervised backbones with a 1x1 dense head. Sizes s, b and l are
published; `g` is the 1.1B teacher and has no LibreYOLO-hosted checkpoint, so
requesting it raises with the reason.

The rest are inference-only ports of earlier published work, each checked
against its pinned upstream source:

- Detection: [DETR](/docs/models/detr),
  [Deformable DETR](/docs/models/deformable-detr),
  [DINO-DETR](/docs/models/dino-detr), [LW-DETR](/docs/models/lw-detr),
  [Faster R-CNN](/docs/models/faster-rcnn), [Mask R-CNN](/docs/models/mask-rcnn),
  [FCOS](/docs/models/fcos), [RetinaNet](/docs/models/retinanet),
  [SSD300](/docs/models/ssd), [CenterNet](/docs/models/centernet) and
  [EfficientDet](/docs/models/efficientdet).
- Classification: [AlexNet](/docs/models/alexnet), [VGG](/docs/models/vgg),
  [ViT](/docs/models/vit), [DeiT](/docs/models/deit) and
  [Swin](/docs/models/swin).
- Semantic segmentation: [FCN](/docs/models/fcn) and
  [DeepLabv3](/docs/models/deeplabv3).
- Depth: [MiDaS](/docs/models/midas). Pose: [HRNet](/docs/models/hrnet),
  top-down on COCO-17 person crops.
- Surface normals: [MoGe-2](/docs/models/moge-2) at 518 px, a DINOv2 patch grid
  with letterbox preprocessing.
- Edge: [TEED](/docs/models/teed) and [DexiNed](/docs/models/dexined), native
  MIT-licensed architectures with local checkpoint converters. Their upstream
  BIPED-trained checkpoints are not bundled, mirrored or auto-downloaded,
  because that dataset's terms are non-commercial.
- Embeddings: [LibreFaceEmbedder](/docs/models/librefacerec), ONNX Runtime
  only.

[LibreMODUS](/docs/models/libremodus) adds 14B-A7B analysis-only inference for
depth, normals, edges and COCO detection from one checkpoint, plus phrase
grounding and image-conditioned `any2any()` chaining.
[LibreFeyNobg](/docs/models/feynobg) adds a matte family at 1024 px, code and
weights Apache-2.0.

[RT-DETRv2](/docs/models/rt-detr) gained
[oriented-object detection](/docs/tasks/oriented-detection) inference for the
official DOTA 1.0 checkpoints in sizes n, s, m, l and x, with
aspect-preserving preprocessing, native `Results.obb` output, validation, and
validated ONNX and TorchScript export.

### Torch-free ONNX inference

`import libreyolo` no longer pulls torch. Model and results names resolve
through a lazy `__getattr__`, and a new top-level `libreyolo.preprocess`
package provides numpy-native preprocessing for deim, deimv2, dfine, ec,
rfdetr, rtdetr, yolo9, yolonas and yolox, so an ONNX Runtime install needs no
torch at all. See [lightweight install](/docs/lightweight-install).

### Export and serving

Five new export formats: [RKNN](/docs/export/rknn) for Rockchip, with `--name`
for the target platform and `--verify` for PC-simulator versus ONNX Runtime
parity; [MNN](/docs/export/mnn); [Paddle](/docs/export/paddle);
[ExecuTorch](/docs/export/executorch); and [Core AI](/docs/export/coreai) on
macOS.

[DeepStream](/docs/export/deepstream) sidecar configs are written by
`deepstream=True` on [ONNX](/docs/export/onnx) export. It is an ONNX-only
option rather than a `format=` key, and is mutually exclusive with `nms=True`.
ExecuTorch and MNN exports now require a checkpoint sidecar,
`<program>.pte.json` and `<model>.mnn.json`.

Two new inference backends: [Triton](/docs/export/triton), with
`create_triton_config` for NVIDIA Triton Inference Server, and Paddle
Inference.

### New CLI commands

`libreyolo enroll` builds an embedding gallery from a folder-per-person tree,
and `libreyolo compare`, aliased as `libreyolo verify`, checks two images by
cosine similarity. See the [CLI reference](/docs/cli).

### Speed

CUDA graph capture of the training step went from 2 families to 24, across
detect, classify, semantic, point and restore. Measured on an RTX 5070 Ti under
AMP: 3.63x on FOMO, 2.74x on MobileNetV4, 1.99x on YOLO9-t, and 1.04x to 1.26x
for the rest, with the win tracking how much of a step is network rather than
loss. End to end on a 20-epoch YOLO9-t fine-tune of 406 images, dataloader and
validation included, 428 s became 368 s with identical mAP50-95 and per-epoch
losses.

Capture at prediction time reaches 39 families, spanning detect, segment, pose,
point, classify, semantic, depth, restore, matte and OCR. Every enabled family
is verified to replay bit-identically against two probe inputs. A family that
cannot be captured whole is split at a verified seam, and the remainder runs
eagerly with identical numbers.

`pip install libreyolo[hub-kernels]` opts every Deformable-DETR-lineage family
into a compiled Apache-2.0 CUDA kernel for multi-scale deformable attention,
pinned to an audited revision. It applies to eager CUDA fp32 only; exports keep
the portable path.

Fused scaled dot product attention now runs across the transformer families
using stock torch, with no optional dependency. Measured on an RTX 5070 Ti
under fp16 autocast, roughly 1.8x on Swin window attention and 3.7x on OWLv2
vision attention. Families held to a byte-exact parity bar keep manual
attention by default and opt in explicitly, and export graphs keep the
primitive-op equation either way.

COCO metrics moved to faster-coco-eval by default: 15.6x faster overall and 56x
on detection-dense datasets, decided on measured parity across all 100 RF100-VL
test splits, where 1381 of 1400 metric values were bit-identical to
pycocotools, the maximum deviation was 2.22e-16, and headline deltas were
exactly 0. `--no-faster-coco-eval` opts out, and pycocotools stays the
automatic fallback when the package is missing.

### Training and validation

Training from scratch (`pretrained=False`) now works for every g0, g1 and g2
family through a seeded random init. Previously only yolo9, rfdetr and dfine
had a scratch path, and the other families silently loaded the pretrained
checkpoint anyway.

Rectangular input arrived: `imgsz=480x640` for prediction and validation
everywhere it makes sense, and rectangular *training* for the seven CNN detect
families, yolo9, yolo9_e2e, yolo9_p2, yolox, yolo7, rtmdet and picodet.
Transformer families and non-detect tasks raise a clear error, and both
dimensions must divide the family stride. Autobatch and DDP spawn honor the
rectangular size when probing.

`amp_dtype` selects bfloat16 for [training](/docs/train) and
[validation](/docs/train/validation), on `TrainConfig`, `ValidationConfig` and
as `--amp-dtype` on `train`, `val` and `profile`. GradScaler is skipped for
bf16. Validation caps are configurable too: `max_det` defaults to 300, and
`eval_max_det` decouples the COCO evaluator cap from NMS.

`val_loss=True` reached every trainable family that can support it, across
detect, classify, semantic and restore rather than the flagships alone. It
moved from `YOLO9Config` and `RFDETRConfig` to `TrainConfig`, so a family that
has not implemented it now raises a clear error instead of ignoring the flag.
Denoising terms are never included, because validation forwards without ground
truth.

Comet, ClearML, Neptune and DVCLive joined the built-in training
[loggers](/docs/train/loggers), with the same canonical metrics and
failure-isolation contract as the existing TensorBoard, MLflow and Weights &
Biases integrations.

The kernel registry moved to `libreyolo/kernels/`, organized by purpose.
`LIBREYOLO_KERNELS` replaces `LIBREYOLO_QUANT_KERNELS`, which is still honored,
and `libreyolo.quant.kernels` remains a working alias.

### Predict sources

Webcams by index, RTSP and HTTP streams, screen capture, and `s3://` and
`gs://` URLs, with `--stream`, `--stream-buffer`, `--vid-stride` and `--show`.
Live sources implicitly enable streaming, which emits one JSON record per
frame. See [prediction sources](/docs/predict/sources).

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
fix carry the old semantics and need the override described on
[upgrading](/docs/upgrade).

Non-strict checkpoint loads now warn with the counts and first names of missing
and unexpected state-dict keys. Shape mismatches always raised, but name
mismatches let a partially matching checkpoint load and then predict with
freshly initialized tensors, leaving no trace.

Auto-conversion no longer risks the file it writes: the new `.pt` is staged and
atomically renamed rather than written in place, and the original file mode is
preserved instead of collapsing to owner-only. Interrupted weight downloads
record a validator for `If-Range` resume and take a cross-process lock, so a
changed remote file or a second process can no longer produce corrupt weights.

CUDA graph capture no longer races with DataLoader pin-memory threads, which
had killed runs with a capture error from the pin-memory thread, and graphs are
captured and replayed on the right device. D-FINE training now applies
upstream's per-size multi-scale recipe instead of a hardcoded
`base_size_repeat=3`.

Several families silently produced wrong geometry and no longer do: PicoDet
clipped boxes against swapped axes, RTMDet segmentation resized masks to a
square derived from width only, Faster R-CNN ONNX exports placed boxes wrong on
non-square images, LW-DETR could drop real detections by running top-K over
unmapped COCO columns, and an exported MoGe-2 model run at an aspect ratio
other than its export canvas stretched the image into wrong normal directions
and now raises instead.

### Contributors

**juni3227**, first-time contributor: rectangular input resolution for the
convolution-based detectors, and a fix for training with YOLO-format datasets
at rectangular model sizes (#649, #658). **Xuban Ceccon**, maintainer:
everything else.

### Release stats

600 commits, 902 files changed, +125,701 / -3,315 lines, 88 merged pull
requests, 8 issues closed. 174 new test modules, 279 to 453. 38 new
documentation pages.

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
