---
title: LingBot-Vision
families: [lingbotvision]
seo_title: "LingBot-Vision: semantic segmentation in LibreYOLO"
description: "Use LingBot-Vision in LibreYOLO for semantic segmentation on an Apache-2.0 ViT backbone. Install, predict, train, validate and export, sizes s/b/l."
lead: "LingBot-Vision is a family of self-supervised vision transformer backbones trained with boundary-centric masked modeling for dense spatial perception, released by Robbyant. LibreYOLO pairs the backbone with a dense head and supports it for one task, semantic segmentation."
keywords: [LingBot-Vision, semantic segmentation, vision transformer, self-supervised pretraining, boundary modeling, Robbyant, dense prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        results = model(SAMPLE_IMAGE, save=True)

        mask = results[0].semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreLingBotVisions-sem.pt source=bus.jpg save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone frozen by default, matching the upstream evaluation
        # protocol: only the 1x1 dense head trains.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Full fine-tune
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].semantic_mask.data.shape)
---

## Install

LingBot-Vision needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`results[0].semantic_mask` carries the dense class map: `.data` is an `(H, W)`
tensor of class IDs on the original image size, and `.classes` lists the class
IDs actually present. `results[0].boxes` is `None`, since there are no
per-instance detections. `conf` and `iou` are accepted for API parity but do
not change the output, since the model returns one class per pixel rather than
detections to filter. See [prediction](/docs/predict) for sources, streaming
and result handling.

## Variants

Three published sizes, s, b and l, distilled from a 1.1B-parameter ViT-g/16
teacher. The teacher itself, size `g`, loads and fine-tunes in LibreYOLO but
LibreYOLO does not host a `g` checkpoint of its own.

<checkpoint-table />

## Train

`train()` fine-tunes a published checkpoint. The default recipe is the
upstream report's linear probe: the ViT backbone is frozen and only the 1x1
dense head trains, matching how the LibreYOLO-hosted weights above were
produced. Pass `freeze_backbone=False` to fine-tune the whole network instead,
and expect to lower `lr0` accordingly.

<code-tabs name="train" />

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys: mIoU and pixel accuracy,
measured against any dataset in the format you trained on.

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

<provenance-box>

The upstream release documents its ViT as built on the DINOv2/DINOv3
architecture published by Meta AI. Robbyant distributes their implementation
under Apache-2.0, and this LibreYOLO port was made only from the Robbyant
repository, never from Meta's DINOv2 or DINOv3 code.

</provenance-box>

## Citation

<citation-block />
