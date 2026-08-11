---
title: Image classification
seo_title: Image classification in LibreYOLO
description: >-
  Label a whole image in LibreYOLO: the families that serve the task, the
  ImageFolder dataset layout, and the predict, train, validate and export calls.
lead: >-
  Image classification assigns one label distribution to a whole image and
  locates nothing inside it. The task key is classify.
keywords:
  - image classification python
  - train image classifier
  - ImageFolder dataset
  - top-1 accuracy
  - zero-shot classification
  - MIT classification library
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -cls suffix in the filename selects the task, so no task
        # argument is needed.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: The whole distribution
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data is the full (C,) vector; top5/top5conf are ordered views.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, no training'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP scores the image against text prompts, so the label set is

        # set at call time instead of baked into the checkpoint.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 is a known dataset name and downloads on first use.
        # Pass a directory with a train/ split for your own data.
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() returns a plain dict, not an object.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like a checkpoint and returns the same Results object.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Definition

Image classification produces one score per class for the whole image and no
coordinates at all. It answers what is in the picture, never where, which is
what separates it from [object detection](/docs/tasks/object-detection).

`classify` is the canonical task key, and the `-cls` suffix in a checkpoint
filename selects it. That suffix is required rather than optional on
classification families, so `LibreResNet50.pt` is not read as a classifier and
only `LibreResNet50-cls.pt` is.

`predict()` fills `result.probs` and leaves `boxes` empty. `.data` is the
full score vector, `.top1` the index of the highest score and `.top1conf` its
value, `.top5` the five highest indices in descending order and `.top5conf`
their scores. Indices point into `result.names`. Slicing a `Results` object
never truncates `probs`, because the vector belongs to the image rather than to
one row.

## Models

Five families both train and predict: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) and
[DINOv2](/docs/models/dinov2). The first four run on the base package and ship
published weights. DINOv2 needs `pip install "libreyolo[rfdetr]"` and has no
LibreYOLO-hosted checkpoint: it loads the upstream backbone with a randomly
initialized linear head, so it is a fine-tuning starting point rather than a
ready predictor.

Five more predict, validate and export, but their `train()` raises
`NotImplementedError`: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) and
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) and [SigLIP2](/docs/models/siglip2) classify without
a fixed label set. They score the image against text prompts, so
`set_classes()` defines the classes at call time and there is no training step
for a new label set at all. Both also serve the `embed` task.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`conf`, `iou` and `max_det` have no effect here: there are no candidates to
threshold or suppress, only one distribution. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Dataset format

Classification uses a directory tree, not label files and not a YAML. `data` is
the dataset root.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` is required for training and it defines the class-to-index mapping by
sorted folder name, so the first folder alphabetically becomes class 0. `val/`
is required for validation. A `test/` split may be present and the default
train and validate commands do not use it. Any split other than `train` has to
contain the same class folder names as the expected class set, which is what
makes a mismatch fail loudly rather than score as a wrong prediction. The
accepted image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`
and `.tiff`.

`data` accepts three things: a path to a directory containing a `train/` split,
a `.zip` URL, or one of the known dataset names, `imagenette160` and `smoke10`,
which download and cache on first use.

The canonical loader is `libreyolo.data.classify_dataset`.

## Train

<code-tabs name="train" />

There is no `nc` to declare: the class count comes from the folder names under
`train/`, and the final linear layer is rebuilt to match it while the backbone
transfers unchanged. See [training](/docs/train) for datasets, augmentation,
multi-GPU and loggers.

## Validate

`val()` returns a plain dictionary of `metrics/` keys, computed over the `val/`
split of the dataset root.

<code-tabs name="val" />

`metrics/accuracy_top1` is the share of images whose highest-scoring class is
the true one, and it is the headline number, the one training uses to pick the
best epoch. `metrics/accuracy_top5` is the share whose true class appears
anywhere in the five highest-scoring classes, which says less the fewer classes
the dataset has. The dictionary also carries `fitness`, a copy of the top-1
value.

## Export

<code-tabs name="export" />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Format coverage differs by family; the matrix on each model page is
generated from the validated set rather than typed by hand. See
[export and deploy](/docs/export) for the formats, their extras and their
constraints.


