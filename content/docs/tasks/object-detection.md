---
title: Object detection
seo_title: "Object detection in LibreYOLO"
description: "Detect objects as axis-aligned boxes in LibreYOLO: the families that serve the task, the label format, and the predict, train, validate and export calls."
lead: "Object detection locates every object instance in an image and returns an axis-aligned rectangle, a class label and a score for each one. The task key is detect."
keywords: [object detection python, detect objects in image, bounding box detection, MIT object detection library, YOLO alternative, train object detector]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(results[0].names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Another family, same call
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the checkpoint, and every detector returns the
        # same Results object, so switching family is a one line change.
        model = LibreYOLO("LibreDFINEn.pt")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy.shape)
    - label: Video and streams
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml downloads a 128-image sample on first use. Point data
        # at your own dataset YAML for a real run.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() returns a plain dict, not an object.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like a checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLO9t.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Definition

Object detection answers where each object is and what it is. One image in,
one row per instance out: four numbers for the rectangle, a class index and a
score. Nothing about pixel shape, orientation or parts is included, which is
what separates it from [instance segmentation](/docs/tasks/instance-segmentation),
[oriented boxes](/docs/tasks/oriented-detection) and
[pose](/docs/tasks/pose-estimation).

`detect` is the canonical task key and the default: a checkpoint whose filename
carries no task suffix loads as a detector.

`predict()` fills `results[0].boxes`. `.xyxy` gives pixel corners on the
original image canvas, `.conf` the score, and `.cls` the class index into
`results[0].names`. `.xywh`, `.xyxyn` and `.xywhn` are derived views of the
same rows, and `.id` carries a track id once a tracker is attached. Iterating
a `Boxes` object yields one-row slices, so `box.cls`, `box.conf` and
`box.xyxy` all work per detection.

## Models

Eleven families both train and predict: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) and [PicoDet](/docs/models/picodet). YOLOv9 and
RF-DETR are the two flagship families, and features land on them first. RF-DETR
needs its own extra, `pip install "libreyolo[rfdetr]"`; the rest run on the
base package.

Eleven more predict, validate and export, but their `train()` raises
`NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) and
[EfficientDet](/docs/models/efficientdet).

The Darknet lineage, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) and
[YOLOv4](/docs/models/yolov4), is kept as a frozen exhibit: predict, validate
and export work, training does not.

A separate group takes its class list at runtime rather than from the
checkpoint, so it detects names never seen in training:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) and [OV-DEIM](/docs/models/ov-deim),
plus the vision-language families
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) and
[LibreMODUS](/docs/models/libremodus). These load through their own factory and
extras; each model page carries the exact call.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`conf` sets the confidence threshold and `max_det` caps the number of rows.
`iou` is the NMS threshold, so it only has an effect on a family that runs NMS;
RF-DETR and the end-to-end YOLOv9 head decode a fixed set of predictions and
ignore it. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Dataset format

One `.txt` label file per image, found by swapping `images` for `labels` in the
image path and changing the extension.

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

Each row is exactly five fields, a class index followed by a normalized
center-and-size box:

```text
<class_id> <cx> <cy> <w> <h>
```

Coordinates are floats in `[0, 1]`, relative to the original image width and
height. `w` and `h` must be positive. A missing or empty label file means the
image has no objects. Rows carry no confidence and no track id.

The YAML names the splits and the classes:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` and `val` may be image directories, image-list `.txt` files, or lists
of either. `nc` is optional and must match `names` when present. Native COCO
JSON works too: add an `annotations` mapping of split name to JSON file, and
the split path then gives the image root. When `names` is present it defines
the label ids, so the JSON category names have to match it.

## Train

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` and `lr0` are the arguments that move first. Learning
rates differ sharply by family: the transformer detectors want `lr0` at or
below `1e-4`, where a convolutional detector tolerates far more. Each model
page states the family's own defaults, and a family can ignore an argument
outright, which its page lists. See [training](/docs/train) for datasets,
augmentation, multi-GPU and loggers.

## Validate

`val()` returns a plain dictionary of `metrics/` keys, computed with COCO
evaluation over the split named by `val` in the dataset YAML.

<code-tabs name="val" />

`metrics/mAP50-95` is mean average precision averaged over IoU thresholds 0.50
to 0.95, and it is the headline number. `metrics/mAP50` and `metrics/mAP75` are
the single-threshold versions. `metrics/precision` and `metrics/recall` are
taken at IoU 0.50. `metrics/mAP_small`, `metrics/mAP_medium` and
`metrics/mAP_large` split the same average by object area, and
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` and `metrics/AR_large` are the matching average-recall
figures. `metrics/AR_max_det` and `metrics/max_det` record the detection cap
the run used. The same numbers repeat under a `(B)` suffix, for box, so that a
detection key reads the same on a model that also predicts masks.

## Export

<code-tabs name="export" />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Format coverage differs by family; the matrix on each model page is
generated from the validated set rather than typed by hand. See
[export and deploy](/docs/export) for the formats, their extras and their
constraints.
