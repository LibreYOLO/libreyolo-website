---
title: Oriented detection
seo_title: "Oriented detection in LibreYOLO"
description: "Detect rotated objects in LibreYOLO: the families that serve oriented boxes, the four-corner label row, and the predict, train, validate and export calls."
lead: "Oriented object detection locates each instance with a rotated rectangle rather than an axis-aligned one, so a tilted object is bounded tightly instead of by a box full of background. The task key is obb."
keywords: [oriented bounding box detection, rotated object detection, OBB python, DOTA dataset, aerial object detection, rotated IoU]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # Needs the rfdetr extra: pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -obb suffix in the filename selects the task, so no task
        # argument is needed.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5): center x, center y, width, height, radians
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Corners instead of angles
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) corner points in pixels
        print(obb.xyxyxyxyn.shape)   # the same, normalized
        print(obb.xyxy.shape)        # (N, 4) enclosing axis-aligned box
    - label: A smaller checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Continues from published oriented weights. data must point at a
        # dataset whose label rows carry four corners.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: From detection weights
      language: bash
      code: |
        # Detection weights carry no angle prediction, so this is an explicit
        # transfer. Asking for task=obb is what authorizes it.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() returns a plain dict, not an object.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like a checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
---

## Definition

Oriented detection adds one number to a detection: the angle. Each instance
gets a rotated rectangle, a class and a score. The gain is tightness. A ship at
45 degrees, a warehouse roof, a row of parked trucks: an axis-aligned box
around any of them is mostly background, and two neighboring boxes overlap even
when the objects do not. That is why the task is standard in aerial imagery and
document layout, and why the reference dataset for it is DOTA.

`obb` is the canonical task key, and the `-obb` suffix in a checkpoint filename
selects it, so `task=` is not needed when loading published weights.

`predict()` fills `result.obb`. `.xywhr` is the canonical `(N, 5)` form:
center x, center y, width, height, and an angle in radians giving the rotation
of the width side around the center. `.conf` and `.cls` carry the score and the
class index into `result.names`, and `.id` a track id when tracking.
`.xyxyxyxy` converts each row to its four corner points as `(N, 4, 2)` pixels,
`.xyxyxyxyn` normalizes those corners, and `.xyxy` gives the enclosing
axis-aligned box, which is what to use when downstream code only understands
rectangles. `result.boxes` is filled as well, with the axis-aligned form.

## Models

[RF-DETR](/docs/models/rf-detr) is the family to use. It trains, predicts,
validates and exports oriented boxes, and it ships published oriented
checkpoints in four sizes, n, s, m and l. It needs its own extra,
`pip install "libreyolo[rfdetr]"`, and its model page carries the weights
license and the provenance.

Read the section below on what those checkpoints actually predict before you
plan around them.

[RT-DETRv2](/docs/models/rt-detr) also declares `obb` in its supported tasks
and can load an oriented checkpoint, but LibreYOLO publishes no oriented
weights for it, so there is nothing to start from.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

Know what the published checkpoints are before you run them. Despite DOTA being
the reference benchmark for this task, these weights were not trained on it.
All four were initialized from the RF-DETR detection weights and fine-tuned on a
single Roboflow Universe dataset of UAV footage, with six vehicle classes: bike,
bus, car, other_vehicle, taxi and truck. Their model cards describe them as
development weights, produced while validating oriented training support, and
say they should not be read as production or benchmark-official weights.

In practice that means they are a working starting point for oriented boxes on
vehicles seen from above, and for verifying that your pipeline runs end to end.
Any other domain, including the aerial categories DOTA is known for, means
training on your own oriented labels. `conf` and `max_det` shape the output as
they do for detection. See [prediction](/docs/predict) for sources, streaming
and result handling.

## Dataset format

The layout is the detection layout: one `.txt` label file per image, found by
swapping `images` for `labels` in the image path and changing the extension.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

A row is exactly nine fields, a class index followed by four corner points in
order:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

The four points are normalized floats in `[0, 1]` and have to form a
non-degenerate oriented rectangle. No angle is stored in the label file: the
loader derives the canonical `xywhr` from the corners. The parser is strict by
default and rejects out-of-range coordinates, while dataset and validation
ingestion may first clip to `[0, 1]` for otherwise valid crop-boundary labels,
then still reject degenerate boxes.

Row parsing is task-aware. Nine fields mean an oriented box only in `obb` mode;
in `segment` mode the same row is read as a four-point polygon.

The YAML is the detection YAML:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

Native COCO JSON loads too, with an `annotations` mapping of split name to JSON
file. Annotations are read in priority order: an `obb` field of eight
pixel-space corners, an `obb` field of `[cx, cy, w, h, angle]` with the angle in
radians, a `segmentation` polygon or RLE refit to its minimum-area rectangle,
or a plain COCO `bbox`, which is treated as an axis-aligned rectangle and
canonicalized to `xywhr`.

The canonical row parser is `libreyolo.data.parse_yolo_obb_label_line`.

## Train

<code-tabs name="train" />

Training continues from a published `-obb` checkpoint by default. Starting from
detection weights is a deliberate transfer: those weights predict no angle, and
passing `task=obb` is what authorizes the swap. Keep `lr0` at or below `1e-4`,
as with the family's other tasks. See [training](/docs/train) for datasets,
augmentation, multi-GPU and loggers.

## Validate

`val()` returns a plain dictionary of `metrics/` keys. Matching uses rotated
IoU, computed between oriented rectangles rather than between their enclosing
axis-aligned boxes, so a prediction with the right position and the wrong angle
scores as a miss.

<code-tabs name="val" />

`metrics/mAP50-95` is mean average precision averaged over IoU thresholds 0.50
to 0.95 in steps of 0.05, and it is the headline number. Unlike the COCO path
used by detection, this task honors `iou_thresholds` in the validation config,
so the sweep can be changed. `metrics/mAP50` and `metrics/mAP75` are the
single-threshold versions. `metrics/precision` and `metrics/recall` are real
precision and recall at IoU 0.50, read at the loosest operating point: every
prediction that survived the confidence threshold is counted, and that
threshold defaults to 0.001 during validation. Raising `conf` therefore moves
them, while the mAP figures, which use the whole precision-recall curve, stay
put. Four of these repeat under an `(OBB)` suffix,
`metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`, `metrics/precision(OBB)` and
`metrics/recall(OBB)`, which is how a caller tells an oriented result from an
axis-aligned one when both sit in the same table. `metrics/mAP75` has no
suffixed twin.

Two options do nothing on this task. `save_json` and `save_plots` are accepted
and log a warning: oriented prediction dumps and validation plots are not
implemented.

## Export

<code-tabs name="export" />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Format coverage differs by task on the same family, and the matrix
on the model page is generated from the validated set and names the reason a
target is unavailable. See [export and deploy](/docs/export) for the formats,
their extras and their constraints.
