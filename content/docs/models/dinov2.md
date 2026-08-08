---
title: DINOv2
families: [dinov2]
seo_title: "DINOv2 in LibreYOLO: semantic, classify and embed"
description: "Use DINOv2 in LibreYOLO for semantic segmentation, classification and whole-image embedding on the DINOv2-with-Registers backbone. Apache-2.0 throughout."
lead: "DINOv2 is a self-supervised vision transformer trained by Meta AI to produce general-purpose image features without labels. LibreYOLO wraps its DINOv2-with-Registers backbone for three tasks: semantic segmentation, classification and whole-image embedding."
keywords: [DINOv2, DINOv2 with registers, self-supervised learning, vision transformer, semantic segmentation, image embedding, feature extraction, Meta AI]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Semantic
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # No LibreYOLO-hosted checkpoint exists for this family: this
        # downloads the Apache-2.0 DINOv2-with-Registers-small backbone from
        # Meta's Hugging Face org. The dense head starts at random
        # initialization until you train it (see Train below).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        results = model(SAMPLE_IMAGE)

        mask = results[0].semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Classify
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= is your dataset's class count; the linear head starts
        # at random initialization until you train it.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        results = model(SAMPLE_IMAGE)

        print(results[0].probs.top1, results[0].probs.top1conf)
    - label: Embed
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Bypasses every task head: the backbone alone is enough, so this
        # needs no fine-tuning to be useful.
        model = LibreDINOv2(size="s", task="embed")
        results = model(SAMPLE_IMAGE)

        print(results[0].embeddings.data.shape)   # (1, D), L2-normalized
    - label: Embed a batch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Convenience wrapper: runs predict() and stacks every row into one
        # (N, D) tensor.
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Semantic
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Classify
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semantic
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Classify
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semantic
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Classify
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embed
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object. Export
        # names the file from the task, here LibreDINOv2s-sem.onnx.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        results = model(SAMPLE_IMAGE)
---

## Install

LibreDINOv2 registers only when `transformers` is installed, the same
optional dependency RF-DETR needs for its DINOv2 backbone, so it needs the
same extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Predict

LibreYOLO does not publish a LibreDINOv2 checkpoint. Construct the wrapper
directly instead of loading a file: `model_path=None` (the default) downloads
Meta's Apache-2.0 `facebook/dinov2-with-registers-small` backbone from Hugging
Face on first use. `task=` selects what runs on top of it.

<code-tabs name="predict" />

`task="semantic"` and `task="classify"` add a dense or linear head on top of
the backbone; that head is randomly initialized and only useful after you
train it (see [Train](#train)). `task="embed"` skips every head and returns
the backbone's final normalized CLS token as one whole-image row in
`results[0].embeddings`, so it needs no training at all. `results[0].boxes` is
always `None`: none of the three tasks produce per-instance detections. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

`size` selects the RF-DETR-style projector width layered on top of the
backbone, not the backbone itself: every size shares the same DINOv2-S
(small) encoder. Semantic segmentation runs at DINOv2's native square patch
grid; classification and embedding run at the smaller classification
resolution used to train the linear probe.

## Train

`task="semantic"` and `task="classify"` both train; `task="embed"` has no
class-dependent head to fit and raises `NotImplementedError` if you call
`train()` on it.

<code-tabs name="train" />

The primary keyword arguments here are `batch_size` and `lr`, not `batch` and
`lr0` used by most other families; `batch` and `lr0` are still accepted and
mapped onto them, but passing both raises a conflict error. `output_dir=`
(default `"runs/train"`) replaces `project=`/`name=` as the primary way to
place a run, though passing `project=`/`name=` directly still works. See
[training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys: mIoU and pixel accuracy for
`task="semantic"`, top-1 and top-5 accuracy for `task="classify"`.
`task="embed"` has no ground truth to score against and raises
`NotImplementedError` if you call `val()` on it.

<code-tabs name="val" />

## Export

<export-matrix />

Each task supports a different subset of formats, shown above. An exported
artifact loads back through `LibreYOLO()` on its file suffix, so a `.onnx` or
`.engine` file behaves like a checkpoint and returns the same `Results`.
[Export](/docs/export) lists the arguments every format accepts.

<code-tabs name="export" />

## Licensing

<provenance-box>

The "Weights" row above names the license that applies, Apache-2.0, but
nothing is actually republished under the LibreYOLO Hugging Face org for this
family: LibreYOLO hosts no LibreDINOv2 checkpoint of its own. What
`LibreDINOv2(model_path=None)` downloads is Meta's own
`facebook/dinov2-with-registers-small` repository, untouched.

</provenance-box>

## Citation

<citation-block />
