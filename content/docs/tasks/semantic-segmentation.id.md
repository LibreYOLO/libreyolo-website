---
title: Semantic segmentation
seo_title: Semantic segmentation in LibreYOLO
description: >-
  Label every pixel with a class in LibreYOLO: the families that serve the task,
  the dense mask format, and the predict, train, validate and export calls.
lead: >-
  Semantic segmentation assigns a class to every pixel of an image and draws no
  distinction between instances of the same class. The task key is semantic.
keywords:
  - semantic segmentation python
  - pixel classification
  - dense prediction
  - train segmentation model
  - mIoU
  - MIT segmentation library
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -sem suffix in the filename selects the task, so no task
        # argument is needed.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) class ids on the original canvas
        print(mask.classes)      # sorted class ids present, ignoring 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: One class at a time
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # boolean (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Another family, same call'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
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
    - label: On ADE20K
      language: bash
      code: |
        # ade20k.yaml carries an embedded download script for the ~1 GB
        # archive, so it needs explicit permission unless the data is local.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() returns a plain dict, not an object.
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
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like a checkpoint and returns the same Results object.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definition

Semantic segmentation labels pixels, not objects. Every pixel receives one
class id, and two cars touching in the image become one region of the car
class with no boundary between them. Counting instances is
[instance segmentation](/docs/tasks/instance-segmentation); labeling every
pixel and separating instances at the same time is
[panoptic segmentation](/docs/tasks/panoptic-segmentation).

`semantic` is the canonical task key, and the `-sem` suffix in a checkpoint
filename selects it, so `task=` is not needed when loading published weights.

`predict()` fills `result.semantic_mask`. `.data` is an `(H, W)` integer
class map on the original image canvas, `.classes` lists the ids present in
sorted order, and `.class_mask(id)` returns the boolean `(H, W)` selection for
one class. The value `255` is the ignore label: it is never a class, it is
excluded from loss and metrics, and `.classes` leaves it out.

## Models

Three families both train and predict:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) and
[DINOv2](/docs/models/dinov2). SegFormer and LingBot-Vision run on the base
package and ship published weights. DINOv2 needs
`pip install "libreyolo[rfdetr]"` and has no LibreYOLO-hosted checkpoint: it
loads the upstream backbone and its dense head starts at random
initialization, so it is a training starting point rather than a ready
predictor.

Four more predict, validate and export, but their `train()` raises
`NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) and
[EoMT](/docs/models/eomt).

Class sets differ by checkpoint, not by family. The published weights come from
datasets whose label spaces have little in common, ADE20K's 150 classes against
Cityscapes' 19 among them, so a checkpoint's `names` is what tells you what it
can label, and two checkpoints are only comparable when they were trained on
the same one.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The map is an argmax per pixel, so there is no NMS step and `iou` never has an
effect. `conf` and `max_det` are accepted for API parity and do nothing on
SegFormer, PIDNet and the other dense predictors; EoMT is the exception, where
`conf` filters query selection. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Dataset format

Each image is paired with a dense single-channel mask rather than a `.txt`
label file, found by swapping `images` for the mask directory in the image path.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Masks are lossless single-channel images, normally PNG, and palette-mode PNGs
are read as palette indices. Each pixel value is a class id in `0..nc-1`, the
value `255` means ignore, and the mask resolution has to equal the paired image
resolution.

The YAML takes two keys on top of the shared contract:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` is the directory name substituted for `images`, defaulting to
`masks`. `label_mapping` is an optional `{source_id: train_id}` remap applied
to mask pixel values at load time, which is how a dataset numbered 1 to 150
becomes 0 to 149; any source value left unmapped becomes ignore, and every
train id has to fall in `0..nc-1`.

Leaving `masks_dir` out switches the loader to a fallback: masks are
rasterized at load time from polygon labels resolved through the usual
`images` to `labels` convention, and a `background` class is appended after the
object classes, so `nc` grows by one.

The canonical loader is `libreyolo.data.SemanticDataset`.

## Train

<code-tabs name="train" />

`imgsz` is constrained here in a way it is not on a detector. Each family
declares a divisor its input has to be a multiple of, set by its patch grid or
output stride, and both training and validation raise a `ValueError` before the
run starts when `imgsz` does not divide evenly. The divisor is 32 for SegFormer,
16 for LingBot-Vision and EoMT, 14 for DINOv2, and 8 for FCN and PIDNet. See
[training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a plain dictionary of `metrics/` keys, computed over the split
named by `val` in the dataset YAML.

<code-tabs name="val" />

`metrics/mIoU` is mean intersection over union: for each class, the overlap
between predicted and true pixels divided by their union, averaged over
classes. It is the headline number and the one used to pick the best epoch
during training. `metrics/pixel_accuracy` is the share of pixels given the
correct class, which a large background class can inflate, so mIoU is the
figure to compare on. Pixels marked `255` count toward neither. The dictionary
also carries `fitness`, a copy of the mIoU value.

## Export

<code-tabs name="export" />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Format coverage differs by family; the matrix on each model page is
generated from the validated set rather than typed by hand. See
[export and deploy](/docs/export) for the formats, their extras and their
constraints.


