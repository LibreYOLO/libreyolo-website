---
title: SegFormer
families: [segformer]
seo_title: "SegFormer: semantic segmentation in LibreYOLO"
description: "Use SegFormer in LibreYOLO for ADE20K semantic segmentation across sizes b0-b5. Install, predict, train and export; pretrained weights are non-commercial."
lead: "SegFormer is a semantic segmentation transformer that pairs a hierarchical Mix Transformer (MiT) encoder with a lightweight all-MLP decode head, avoiding the heavy decoders and fixed positional encodings earlier segmentation transformers needed. LibreYOLO supports it for one task, semantic segmentation, across six sizes."
keywords: [SegFormer, semantic segmentation, Mix Transformer, MiT, transformer segmentation, ADE20K, dense prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt source=bus.jpg save=True
  train:
    - label: Python (fine-tune)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: From scratch
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # No model_path: random init, nothing downloaded. The only route to
        # weights free of the pretrained checkpoints' non-commercial term.
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## Install

SegFormer needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`result.semantic_mask` carries the dense class map: `.data` is an `(H, W)`
tensor of class IDs on the original image size, and `.classes` lists the class
IDs actually present. `result.boxes` is `None`, since there are no
per-instance detections. `conf` and `iou` are accepted for API parity but do
not change the output: the model returns one class per pixel, not per-instance
detections to filter or de-duplicate. See [prediction](/docs/predict) for
sources, streaming and result handling.

## Variants

Six sizes, b0 through b5, widening and deepening the Mix Transformer encoder
at each step while keeping the same all-MLP decode head design.

<checkpoint-table />

## Train

`train()` fine-tunes a published checkpoint by default. Pass no `model_path`
to `LibreSegformer(...)` instead and it builds with a randomly initialized
encoder and head, training from scratch, the only route to weights that carry
none of the pretrained checkpoints' non-commercial restriction (see
[Licensing](#licensing)).

<code-tabs name="train" />

Left alone, the trainer follows the SegFormer paper's ADE20K recipe: AdamW at
a backbone base learning rate with the decode head trained at 10x that rate,
weight decay everywhere except LayerNorm and the Mix-FFN positional
convolution, and a linear decay schedule with a warmup. Convergence for the
larger sizes, b3 through b5, has not been validated end to end.

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

LibreSegformer's encoder and decode head are a PyTorch port of Hugging Face
Transformers' Apache-2.0 SegFormer implementation, not of NVlabs/SegFormer:
NVIDIA's original repository was never read or copied, and is credited here
only for attribution to the paper's authors. Only the pretrained checkpoints
above carry NVIDIA's non-commercial restriction; the architecture and
LibreYOLO's own code stay MIT throughout.

</provenance-box>

## Citation

<citation-block />
