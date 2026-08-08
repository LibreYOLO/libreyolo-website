---
title: MobileNetV4
families: [mobilenetv4]
seo_title: "MobileNetV4: train, validate and export under Apache-2.0"
description: "Use MobileNetV4 in LibreYOLO for image classification. Install, predict, fine-tune, validate and export LibreMobileNetV4 small/medium/large."
lead: "MobileNetV4 is an image classifier built for mobile and edge hardware, using the Universal Inverted Bottleneck block to unify several prior mobile block designs into one searchable structure. LibreYOLO supports it for one task: classification."
keywords: [MobileNetV4, MobileNetV4 conv, image classification, mobile inference, edge classifier, ImageNet classifier]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx
        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Install

MobileNetV4 needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different model is a one line change. A classifier carries no boxes or
masks: `result.probs` holds the whole-image prediction, with `top1`, `top5`,
`top1conf` and `top5conf`. `conf`, `iou` and `max_det` are accepted for API
parity but have no effect, since there is nothing to threshold or suppress on
a single probability vector. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Three sizes, small/medium/large, all conv-only: this family excludes the
hybrid variants that add Mobile MQA attention. Picking a size is a straight
parameter-count-for-accuracy trade. The task is fixed: every size covers
classification only. The weights filename ends `-cls.pt` on every size, and
that suffix is what the factory reads to route to this family; no `task=`
argument is needed.

## Train

Fine-tuning starts from the published ImageNet backbone and rebuilds the
final classifier layer to the target dataset's class count automatically.

<code-tabs name="train" />

Left alone, the trainer runs 100 epochs at `lr0=1e-3` with AdamW, a batch of
64 and early stopping after 50 epochs without improvement. `data` accepts a
dataset root (`train/` and `val/`, one folder per class), a known short name
such as `imagenette160`, or a `.zip` URL. `lora=True` is not supported here;
passing it raises, since LoRA in LibreYOLO targets transformer components
with `nn.Linear` layers and this family's UIB blocks have none.

See [training](/docs/train) for datasets, augmentation, multi-GPU and
loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys. For classification that is
top-1 and top-5 accuracy over the validation split.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so
a `.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. [Export](/docs/export) lists the arguments every format accepts
and the extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
