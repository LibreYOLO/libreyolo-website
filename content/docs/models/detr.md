---
title: DETR
families: [detr]
seo_title: "DETR: predict and export under Apache-2.0"
description: "Run DETR, the original detection transformer, in LibreYOLO. Install, predict, validate and export four ResNet-based sizes, all Apache-2.0 licensed."
lead: "DETR is the original detection transformer, predicting a fixed set of objects with a Hungarian-matched transformer decoder instead of anchors or a dense grid. LibreYOLO ships four sizes for detection, inference only."
keywords: [DETR, detection transformer, object detection, Hungarian matching, transformer decoder, Meta AI, Facebook AI Research]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDETRr50.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800
        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

DETR needs no optional extra. Everything it imports is in the base install.

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

DETR is inference-only in LibreYOLO. Upstream trains for 500 epochs with
Hungarian matching; that recipe is not implemented here, so `train()` raises
`NotImplementedError`.

## Variants

Four checkpoints combine two backbone depths, ResNet-50 or ResNet-101, with an
optional dilated C5 stage: the DC5 variants keep the last backbone stage at
full resolution instead of downsampling further, so the decoder reads a finer
feature map from the same input size. All four share 100 learned object
queries and a six-layer transformer encoder-decoder, and all run at the same
input resolution.

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

The authors publish no BibTeX block for this work. Cite the paper linked in
the Upstream row above.
