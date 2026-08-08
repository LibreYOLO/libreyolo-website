---
title: Deformable DETR
families: [deformable_detr]
seo_title: "Deformable DETR: predict and export, Apache-2.0"
description: "Run Deformable DETR in LibreYOLO for object detection. Install, predict, validate and export five sparse-attention sizes, all Apache-2.0 licensed."
lead: "Deformable DETR replaces DETR's dense cross-attention with sparse, multi-scale sampling around each reference point, which is what made transformer detectors practical to train. LibreYOLO ships five sizes for detection, inference only."
keywords: [Deformable DETR, detection transformer, sparse attention, multi-scale attention, object detection, SenseTime]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeformableDETRr50.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800
        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt imgsz=800 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

Deformable DETR needs no optional extra. Everything it imports is in the base
install, using a pure-PyTorch multi-scale deformable attention core.

```bash
pip install libreyolo
```

Installing `libreyolo[hub-kernels]` is optional. Once the `kernels` package is
present, LibreYOLO fetches a compiled multi-scale deformable attention kernel
from the Hugging Face Hub at runtime and uses it in place of the pure-PyTorch
core; `LIBREYOLO_HUB_KERNELS=0` turns it back off.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the query
selection; `iou` is accepted for API parity but has no effect, because the
decoder is a set predictor with no NMS step. See
[prediction](/docs/predict) for sources, streaming and result handling.

Deformable DETR is inference-only in LibreYOLO. Upstream trains with Hungarian
matching and a focal classification loss; that recipe is not implemented here,
so `train()` raises `NotImplementedError`.

## Variants

Five checkpoints cover the released configurations, all at the same input
resolution. `r50ss` restricts attention to a single feature scale; `r50ssdc5`
adds a dilated C5 backbone stage on top of that. `r50` is the default
multi-scale configuration, sampling across four feature-map levels.
`r50refine` adds iterative bounding box refinement across decoder layers, and
`r50twostage` generates its initial region proposals from the encoder output
instead of learned queries.

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
