---
title: Instance segmentation
seo_title: Instance segmentation in LibreYOLO
description: >-
  Segment individual objects in LibreYOLO: the families that serve the task, the
  polygon label format, and the predict, train, validate and export calls.
lead: >-
  Instance segmentation locates every object instance and returns a per-pixel
  mask for each one, alongside the box, class and score a detector returns. The
  task key is segment.
keywords:
  - instance segmentation python
  - object mask prediction
  - segmentation model training
  - polygon labels
  - MIT segmentation library
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -seg suffix in the filename selects the mask head, so no task
        # argument is needed.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), one mask per detection
        print(result.boxes.xyxy.shape)   # (N, 4), the same N rows
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Mask outlines
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xy is a list of (P, 2) contours in pixels, .xyn the same normalized.
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Another family, same call'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continues from published segmentation weights, mask head included.

        # data must point at a dataset whose labels carry polygons.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: From detection weights
      language: bash
      code: |
        # Detection weights carry no mask head, so this is an explicit
        # transfer: the head starts untrained. Asking for task=segment is
        # what authorizes it.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # masks
        print(metrics["metrics/mAP50-95(M)"])    # masks, explicit
        print(metrics["metrics/mAP50-95(B)"])    # boxes
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like a checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Definition

Instance segmentation is detection plus shape. Each object instance still gets
a box, a class and a score, and it also gets a binary mask covering the pixels
that belong to it. Masks may overlap, and pixels belonging to no object are
left unassigned, which is what separates the task from
[semantic segmentation](/docs/tasks/semantic-segmentation) and
[panoptic segmentation](/docs/tasks/panoptic-segmentation).

`segment` is the canonical task key, and the `-seg` suffix in a checkpoint
filename selects it, so `task=` is not needed when loading published weights.

`predict()` fills `result.masks` alongside `result.boxes`. `.data` is
an `(N, H, W)` stack on the original image canvas, row-aligned with the boxes,
so mask `i` belongs to box `i`. `.xy` converts each mask to its largest outer
contour as a `(P, 2)` pixel array, and `.xyn` gives the same contour
normalized.

## Models

Four families both train and predict masks: [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) and
[RTMDet](/docs/models/rtmdet). RF-DETR needs its own extra,
`pip install "libreyolo[rfdetr]"`; the other three run on the base package.

[Mask R-CNN](/docs/models/mask-rcnn) predicts, validates and exports masks, but
its `train()` raises `NotImplementedError`.

[EoMT](/docs/models/eomt) predicts and validates masks and also cannot train,
and its export is narrower still: `export()` only accepts the semantic task, and
raises `NotImplementedError` for `segment` and `panoptic`, because the
query-mask runtime contract those two need has not been defined. Use EoMT for
instance masks in Python, not through an exported graph.

A separate group segments from a prompt rather than a class list: a click, a
box or a phrase picks the object, and the model returns its mask.
[SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) and [PicoSAM3](/docs/models/picosam3) work this
way, as does [SenseNova-Vision](/docs/models/sensenova-vision), whose
segmentation is referring: it takes a phrase naming one object. They load
through their own factory and extras, and each model page carries the exact
call.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`conf` and `max_det` shape the output the same way they do for detection, and
masks are filtered along with the boxes they belong to. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Dataset format

The layout is the detection layout: one `.txt` label file per image, found by
swapping `images` for `labels` in the image path and changing the extension.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

What changes is the row. A segment is a class index followed by a flat polygon:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

At least three points, so the coordinate count after the class index is even
and at least six, and the polygon must be non-degenerate. Coordinates are
floats in `[0, 1]` relative to the original image width and height. A five
field detection row is also accepted in a segmentation dataset and is read as a
rectangular segment, which makes a box-only dataset loadable without a
conversion pass.

The YAML is the detection YAML:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Native COCO JSON works as well: add an `annotations` mapping of split name to
JSON file, and the split path gives the image root.

## Train

<code-tabs name="train" />

Training continues from a published `-seg` checkpoint by default. Starting from
detection weights is possible but is a deliberate transfer: those weights carry
no mask head, so it starts untrained, and passing `task=segment` is what
authorizes the swap. See [training](/docs/train) for datasets, augmentation,
multi-GPU and loggers.

## Validate

`val()` returns a plain dictionary of `metrics/` keys. Boxes and masks are
scored separately, both with COCO evaluation, and the mask numbers are the
primary ones.

<code-tabs name="val" />

The unsuffixed keys hold mask results: `metrics/mAP50-95`, `metrics/mAP50`,
`metrics/mAP75`, then `metrics/mAP_small`, `metrics/mAP_medium` and
`metrics/mAP_large` by object area, and `metrics/AR1`, `metrics/AR10`,
`metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium`, `metrics/AR_large`
for average recall. `metrics/AR_max_det` and `metrics/max_det` record the
detection cap the run used.

Four figures are also published under an explicit suffix, `(M)` for mask and
`(B)` for box, so that a comparison never depends on which number the family
decided to call primary: `metrics/mAP50-95(M)` and `metrics/mAP50-95(B)`,
`metrics/mAP50(M)` and `metrics/mAP50(B)`, `metrics/precision(M)` and
`metrics/precision(B)`, `metrics/recall(M)` and `metrics/recall(B)`. There is
no unsuffixed `metrics/precision` or `metrics/recall` on this task.

Read the precision and recall keys carefully. They are kept for backward
compatibility and are aliases, not an operating point: `metrics/precision(M)`
holds the same value as `metrics/mAP50-95(M)`, and `metrics/recall(M)` the same
value as mask AR at 100 detections, with `(B)` behaving the same way for boxes.
Plotting a pair of them reports one number twice.

## Export

<code-tabs name="export" />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Segmentation coverage is narrower than detection coverage on the
same family. The matrix on each model page is generated from the validated set
and names the reason a target is unavailable. See
[export and deploy](/docs/export) for the formats, their extras and their
constraints.


