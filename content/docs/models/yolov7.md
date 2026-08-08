---
title: YOLOv7
families: [yolo7]
seo_title: "YOLOv7 in LibreYOLO: predict, train and export under MIT"
description: "Run YOLOv7 in LibreYOLO for object detection: install, predict, train, validate and export, MIT-licensed code and weights."
lead: "YOLOv7 is an anchor-based, single-stage detector whose head adds learned implicit-knowledge offsets before the final convolution. LibreYOLO supports its single published size for detection."
keywords: [YOLOv7, object detection, anchor-based detection, implicit knowledge, ImplicitA, real-time object detection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO7b.pt source=bus.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16, lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Warm start from a fresh model
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True always loads the published LibreYOLO7b.pt checkpoint,
        # regardless of what this instance was constructed with. Constructing
        # the class directly, rather than through LibreYOLO(), starts with no
        # weights loaded at all.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640
        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

YOLOv7 needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` sets the confidence threshold
and `iou` the NMS threshold applied after the anchor-based head is decoded. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

LibreYOLO ships one size, `b`. Upstream publishes a single YOLOv7 model, so
there is no size to choose between.

## Train

<code-tabs name="train" />

`pretrained` is read, unlike the no-op of the same name on some other
families here: pass `True` to warm-start from the published `LibreYOLO7b.pt`
checkpoint (auto-downloaded), or a path or name for anything else. That
published checkpoint is 80-class COCO, so requesting it on a model already
rebuilt for a different class count first rebuilds back to 80, loads it, then
transfers every shape-matching tensor into the target head count once the
dataset's class count is read. `resume=True` cannot be combined with
`pretrained`. Left at the default `None`, training continues from whatever
the model was constructed with, or from a random initialization if nothing
was loaded.

Left alone otherwise, the trainer runs 300 epochs at `lr0=0.01` with SGD
momentum 0.937, a 3-epoch warmup, and the same SimOTA assignment and final
15-epoch no-augmentation phase YOLOX uses, adapted to the anchor-based head.
The one difference: YOLOX adds an L1 box-regression refinement during those
final epochs that v7 skips, because v7's SimOTA loss carries no raw-offset L1
branch to refine.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
