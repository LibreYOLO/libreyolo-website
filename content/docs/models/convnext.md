---
title: ConvNeXt
families: [convnext]
seo_title: "ConvNeXt: train, validate and export under Apache-2.0"
description: "Use ConvNeXt in LibreYOLO for image classification. Install, predict, fine-tune with LoRA, validate and export LibreConvNeXt tiny/small/base."
lead: "ConvNeXt is an image classifier built entirely from standard convolutions, modernized block by block from a ResNet toward the design choices of a vision transformer. LibreYOLO supports it for one task: classification."
keywords: [ConvNeXt, ConvNeXt tiny, image classification, pure convolutional network, ImageNet classifier]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Install

ConvNeXt needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

Adapter fine-tuning with `lora=True` is the exception, and needs the `lora`
extra.

```bash
pip install "libreyolo[lora]"
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

Three sizes, tiny/small/base, all trained and evaluated the same way, so
picking one is a straight parameter-count-for-accuracy trade. The task is
fixed: every size covers classification only. The weights filename ends
`-cls.pt` on every size, and that suffix is what the factory reads to route
to this family; no `task=` argument is needed.

## Train

Fine-tuning starts from the published ImageNet backbone and rebuilds the
final classifier layer to the target dataset's class count automatically.

<code-tabs name="train" />

Left alone, the trainer runs 100 epochs at `lr0=1e-3` with AdamW, a batch of
64 and early stopping after 50 epochs without improvement. `data` accepts a
dataset root (`train/` and `val/`, one folder per class), a known short name
such as `imagenette160`, or a `.zip` URL. ConvNeXt's blocks carry the
`nn.Linear` MLPs LoRA needs, so `lora=True` is supported here, and injects
adapters into the block MLPs rather than fine-tuning the full backbone.

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

<provenance-box>

Only ConvNeXt V1 is shipped in this family. ConvNeXt-V2's small pretrained
checkpoints are CC-BY-NC 4.0 and are deliberately excluded, since a
non-commercial weight cannot be redistributed inside an MIT/commercial
library.

</provenance-box>

## Citation

<citation-block />
