---
title: LW-DETR
families: [lwdetr]
seo_title: "LW-DETR: predict and export under Apache-2.0"
description: "Run LW-DETR in LibreYOLO for real-time object detection. Install, predict, validate and export five ViT-based sizes, all Apache-2.0 licensed."
lead: "A plain-ViT detection transformer that Baidu positioned as a real-time alternative to YOLO detectors. LibreYOLO ships five sizes for detection, inference only."
keywords: [LW-DETR, detection transformer, real-time object detection, plain ViT, DETR, Baidu, Atten4Vis]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreLWDETRt.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640
        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreLWDETRt.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

LW-DETR needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the query
selection; `iou` is accepted for API parity but has no effect, because the
decoder is a set predictor with no NMS step. See
[prediction](/docs/predict) for sources, streaming and result handling.

LW-DETR is inference-only in LibreYOLO. Upstream trains with Group-DETR
one-to-many supervision across multiple query groups and an IoU-aware
classification loss; that recipe is not wired here, so `train()` raises
`NotImplementedError`.

## Variants

Five sizes, all sharing the plain-ViT encoder, multi-scale projector and
deformable DETR decoder, and all running at the same input resolution. The two
smallest share an encoder width and split by block depth; the next two share a
wider encoder and split by how many projector levels feed the decoder; the
largest steps up to the widest encoder.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. [Export](/docs/export) lists the arguments every format accepts.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
