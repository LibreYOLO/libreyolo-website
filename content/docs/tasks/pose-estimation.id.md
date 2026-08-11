---
title: Pose estimation
seo_title: Pose estimation in LibreYOLO
description: >-
  Predict keypoints per instance in LibreYOLO: the families that serve the task,
  the label format, and the predict, train, validate and export calls.
lead: >-
  Pose estimation locates each instance and returns an ordered set of named
  keypoints for it, so the output carries the object's internal structure rather
  than only its extent. The task key is pose.
keywords:
  - pose estimation python
  - keypoint detection
  - human pose model
  - COCO keypoints
  - OKS mAP
  - train pose model
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -pose suffix in the filename selects the keypoint head, so no
        # task argument is needed.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) pixel coordinates
        print(result.boxes.xyxy.shape)     # (N, 4), the same N instances
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Visible keypoints only
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible is derived from the third keypoint column, and is
        # all-true when the checkpoint predicts only (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Top-down instead
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNet is top-down: it crops each person first. With no person source
        # given it pairs itself with a LibreYOLO9t detector and logs the choice.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml carries an embedded download script, so it needs
        # explicit permission unless the data is already local.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Your own dataset
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml must declare kpt_shape, and the label rows must carry
        # exactly 5 + K * D fields.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() returns a plain dict, not an object.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like a checkpoint and returns the same Results object.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Definition

Pose estimation returns structure, not just extent. Each instance still gets a
box, a class and a score, and it also gets `K` keypoints in a fixed order, so
index 5 means the same body part on every instance and in every image. The
label set defines that order; nothing in the output identifies a keypoint by
name.

`pose` is the canonical task key, and the `-pose` suffix in a checkpoint
filename selects it, so `task=` is not needed when loading published weights.

`predict()` fills `result.keypoints` alongside `result.boxes`. `.data`
is `(N, K, 2)` or `(N, K, 3)`, row-aligned with the boxes, so instance `i` in
one is instance `i` in the other. `.xy` slices the pixel coordinates and `.xyn`
normalizes them by the original image size. `.conf` is the third column when
the checkpoint predicts one and `None` when it does not, and `.has_visible` is
the boolean mask derived from it, all-true when there is no third column.

Two architectures reach this output. A one-stage model predicts boxes and
keypoints in a single pass. A top-down model runs a detector first, crops each
instance and regresses keypoints inside the crop, so its accuracy depends on
the detector in front of it.

## Models

Three families both train and predict:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) and
[YOLO-NAS](/docs/models/yolo-nas), all one-stage. RF-DETR needs its own extra,
`pip install "libreyolo[rfdetr]"`. RF-DETR and EdgeCrafter ship published pose
checkpoints and both fine-tune on single-class, person-only datasets;
EdgeCrafter's keypoint head is fixed at construction and rejects a dataset
declaring a different count, while RF-DETR reinitializes its head for one. YOLO-NAS
pulls its weights from Deci.AI's own CDN under a non-commercial license, and
LibreYOLO publishes none of them; its pose head also rebuilds for a new
keypoint count, and it is the only one of the three whose class count is not
fixed at one, so it is the family for a multi-class or non-human skeleton, such
as animal pose.

[HRNet](/docs/models/hrnet) is the top-down option. It predicts, validates and
exports, and its `train()` raises `NotImplementedError`. Given no person
source, it pairs itself with a LibreYOLO9t detector automatically; `cropped=True`
treats the whole image as one instance, `person_boxes=` takes boxes you already
have, and `person_detector=` names a different detector.

[SenseNova-Vision](/docs/models/sensenova-vision) also emits keypoints. It is a
prompted generative model with its own factory, `LibreVLM`, and its own extra;
with no vocabulary set, `set_task("pose")` falls back to the person category.
Its weights are non-commercial, and per-image latency is far higher than a
purpose-built pose head, because every prediction is a diffusion decode.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

Keypoint counts and orders are properties of the checkpoint, not of the
library, so a model trained on a different skeleton returns a different `K` and
a different meaning per index. What the third keypoint column holds is also a
checkpoint property: EdgeCrafter writes a constant there rather than a
per-point score, and it has no box head at all, so each of its pose boxes is
the bounding extent of that instance's own keypoints. See
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

A row is a detection row with the keypoints appended:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

The field count is exactly `5 + K * D`, where `D` is the second value of
`kpt_shape`. Box and keypoint coordinates are normalized floats relative to the
original image width and height. Visibility `v`, present only when `D` is 3, is
`0`, `1` or `2`.

The YAML adds two keys to the shared contract:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` is required and is `[K, 2]` or `[K, 3]`. `flip_idx` is optional and
is a permutation of `0..K-1` giving, for each keypoint, the index it takes
after a horizontal flip, which is how a left wrist stays a left wrist. Omit it
and horizontal flip augmentation is switched off for keypoints rather than
applied with the wrong index order.

## Train

<code-tabs name="train" />

Training continues from a published `-pose` checkpoint, which already carries
the keypoint head; the task is read from the checkpoint you load, not a flag
passed at train time, so a detection checkpoint does not become a pose run by
asking for one. `kpt_shape` in your YAML has to match the head exactly for
EdgeCrafter, since its head is fixed at construction, while RF-DETR and
YOLO-NAS resize the head for a different count instead. See
[training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a plain dictionary of `metrics/` keys. Scoring is COCO keypoint
evaluation over Object Keypoint Similarity, which weighs each keypoint's
distance error by the instance scale and by a per-keypoint tolerance, so it
plays the role IoU plays for boxes. It needs `pycocotools`, which is in the
base install.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` is the headline number, mean average precision
averaged over OKS thresholds 0.50 to 0.95, and it is what training uses to pick
the best epoch. `metrics/keypoints_mAP50` and `metrics/keypoints_mAP75` are the
single-threshold versions, and `metrics/keypoints_mAP_M` and
`metrics/keypoints_mAP_L` split the average by instance area, medium and large;
COCO keypoint evaluation defines no small bucket. The matching average recall
figures are `metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` and
`metrics/keypoints_AR_L`. Every key on this task is prefixed `keypoints_`, so
the box `mAP` keys a detector returns do not appear.

## Export

<code-tabs name="export" />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Format coverage differs by family; the matrix on each model page is
generated from the validated set rather than typed by hand. See
[export and deploy](/docs/export) for the formats, their extras and their
constraints.


