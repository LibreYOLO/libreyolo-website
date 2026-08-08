---
title: CLIP
families: [clip]
seo_title: "CLIP in LibreYOLO: zero-shot classify and embed"
description: "Use CLIP in LibreYOLO for zero-shot image classification and image/text embedding. No training: set_classes() defines the label set at runtime."
lead: "CLIP is a dual-tower model that scores an image against text prompts instead of a fixed label set. LibreYOLO supports it for zero-shot classification and image/text embedding, with no training step."
keywords: [CLIP, OpenCLIP, zero-shot classification, image embedding, text embedding, open vocabulary, LAION-2B]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: |
        # With no set_classes() call, CLI predict uses the 1,000 ImageNet
        # class names the model loads with by default.
        libreyolo predict model=LibreCLIPb32-cls.pt source=bus.jpg save=True
    - label: Image and text embedding
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # Both are L2-normalized, so a plain dot product is cosine similarity.
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data is an ImageFolder root with a train/ split; its folder names
        # become the zero-shot class prompts for this run.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # The current set_classes() labels and the input resolution are baked
        # into the graph. Re-export after changing either one.
    - label: CLI
      language: bash
      code: |
        # No set_classes() call here, so this bakes in the default 1,000
        # ImageNet classes the model loads with.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Embedding export
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" traces the image tower alone; no classes needed.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
---

## Install

CLIP needs its own extra, which pulls in the packages its vendored BPE tokenizer uses to reproduce exact token ids.

```bash
pip install "libreyolo[clip]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`set_classes()` is the one primitive that makes this an open-vocabulary classifier: it renders each label into every prompt template, encodes and averages the results, and caches the resulting `[K, D]` matrix as the classifier head, so it is not recomputed per image. Call it again to change classes at any time. With no call, LibreCLIP loads with the 1,000 ImageNet-1k class names already set.

With `task="embed"`, prediction returns one L2-normalized image vector per input instead of class probabilities, and `embed_text()` returns normalized text rows in the same vector space, so a plain dot product between them is cosine similarity. `iou` has no effect on either task; there is no NMS step. See [prediction](/docs/predict) for sources, streaming and result handling.

## Validate

`val()` reads the class-folder names under an ImageFolder `train/` split, calls `set_classes()` with them, then measures zero-shot top-1 and top-5 accuracy. Accuracy depends on how the class names read as prompts, not on any weight update, since there is nothing to train. Validation only covers `task="classify"`; `task="embed"` has no dataset validator.

<code-tabs name="val" />

## Export

<export-matrix />

Export bakes the model's current state into a fixed graph. For `task="classify"`, whatever labels `set_classes()` last set, and the resolution at export time, are baked into a final linear layer, so the exported ONNX or TensorRT graph is an ordinary `[B, K]` image classifier with no text tower and no tokenizer; export again after changing either the classes or the size. `task="embed"` export traces the image tower alone. Both need ONNX opset 14 or higher, which the exporter sets by default.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family. Both are converted from OpenCLIP's LAION-2B-trained checkpoints (`ViT-B-32` and `ViT-B-16`), not from any COCO training run.

<checkpoint-table />

The LAION-2B training data has a documented history of CSAM content (Stanford Internet Observatory, December 2023). LAION has since released Re-LAION, a cleaned re-release; prefer Re-LAION-derived checkpoints where available if you re-host these weights further.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
