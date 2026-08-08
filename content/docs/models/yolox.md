---
title: YOLOX
families: [yolox]
seo_title: "YOLOX: predict, train and export under Apache-2.0"
description: "Use YOLOX in LibreYOLO for object detection: install, predict, train, validate and export under Apache-2.0."
lead: "YOLOX is an anchor-free, single-stage detector with a decoupled classification-regression head, trained with SimOTA label assignment. LibreYOLO supports it for detection."
keywords: [YOLOX, object detection, anchor-free detection, decoupled head, SimOTA, real-time object detection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLOXs.pt source=bus.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16, lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Against COCO
      language: bash
      code: |
        # The bundled COCO yaml carries an embedded download script, so it
        # needs explicit permission unless the dataset is already local.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640
        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLOXs.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

YOLOX needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` sets the confidence threshold
and `iou` the NMS threshold applied across the three decoupled prediction
scales. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Variants

Six sizes share the same CSP backbone and PAFPN neck. The two smallest, `n`
and `t`, run at a smaller fixed input resolution than the other four; the
benchmark table below carries the exact figure for each.

<benchmark-table task="detect" />

<va-embed />

## Train

<code-tabs name="train" />

Left alone, the trainer runs 300 epochs at `lr0=0.01` with SGD momentum 0.9, a
5-epoch warmup and mosaic and mixup augmentation switched off for the final 15
epochs. `train()` also accepts a `pretrained` argument, but the value is never
read inside the method: training always continues from whatever weights the
model was constructed with, so `pretrained=False` does not reinitialize the
network.

`imgsz` defaults to a fixed value in the base training config, not to the
loaded checkpoint's native resolution. That affects the `n` and `t`
checkpoints specifically: continuing to train either one without setting
`imgsz` explicitly switches it up to the larger default rather than the
smaller size it was published at.

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
A CoreML export can bake NMS into the graph with `nms=True`; YOLOX and YOLOv9
are the only two families that flag currently accepts.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
