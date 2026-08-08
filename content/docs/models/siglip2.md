---
title: SigLIP2
families: [siglip2]
seo_title: "SigLIP2 in LibreYOLO: zero-shot classify and embed"
description: "Use SigLIP2 in LibreYOLO for zero-shot image classification and image/text embedding, with sigmoid multi-label scoring. No training needed."
lead: "SigLIP2 is a dual-tower model that scores an image against text prompts with an independent sigmoid per class, instead of a shared softmax over a fixed label set. LibreYOLO supports it for zero-shot classification and image/text embedding, with no training step."
keywords: [SigLIP2, SigLIP 2, zero-shot classification, image embedding, text embedding, open vocabulary, multilingual, sigmoid loss]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        results = model(SAMPLE_IMAGE, save=True)

        r = results[0]
        print(model.names[r.probs.top1], float(r.probs.top1conf))
    - label: CLI
      language: bash
      code: |
        # With no set_classes() call, CLI predict uses the 1,000 ImageNet
        # class names the model loads with by default.
        libreyolo predict model=LibreSigLIP2b16-cls.pt source=bus.jpg save=True
    - label: Multi-label sigmoid scoring
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)[0]

        # Independent per-class probabilities: more than one, or none, can
        # score high at once. Softmax (the default) instead normalizes them
        # into a single-label distribution, matching LibreCLIP's behavior.
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Image and text embedding
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE)[0].embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # Both are L2-normalized, so a plain dot product is cosine similarity.
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data is an ImageFolder root with a train/ split; its folder names
        # become the zero-shot class prompts for this run.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # The current set_classes() labels and the input resolution are baked
        # into the graph. Re-export after changing either one. multi_label
        # must be False (the default) at export time.
    - label: CLI
      language: bash
      code: |
        # No set_classes() call here, so this bakes in the default 1,000
        # ImageNet classes the model loads with.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Embedding export
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" traces the image tower alone; no classes needed.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
---

## Install

SigLIP2 needs its own extra, which pulls in the SentencePiece package its multilingual tokenizer uses.

```bash
pip install "libreyolo[siglip2]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`set_classes()` is the one primitive that makes this an open-vocabulary classifier: it renders each label into every prompt template, encodes and averages the results, and caches the resulting `[K, D]` matrix as the classifier head, so it is not recomputed per image. Call it again to change classes at any time. With no call, LibreSigLIP2 loads with the 1,000 ImageNet-1k class names already set.

SigLIP scores each class independently: `logit = scale * (image . text) + bias`. By default that logit set is still passed through a softmax, giving a single-label distribution that matches LibreCLIP's `top1`/`top5` behavior. Passing `multi_label=True` to `set_classes()` (or at construction) switches to independent sigmoid probabilities instead, so more than one class, or none, can score high on the same image. The tokenizer is a multilingual SentencePiece model (Gemma vocabulary), so class names in languages other than English work the same way.

With `task="embed"`, prediction returns one L2-normalized image vector per input instead of class probabilities, and `embed_text()` returns normalized text rows in the same vector space, so a plain dot product between them is cosine similarity. `iou` has no effect on either task; there is no NMS step. See [prediction](/docs/predict) for sources, streaming and result handling.

## Validate

`val()` reads the class-folder names under an ImageFolder `train/` split, calls `set_classes()` with them, then measures zero-shot top-1 and top-5 accuracy under softmax scoring. Accuracy depends on how the class names read as prompts, not on any weight update, since there is nothing to train. Validation only covers `task="classify"`; `task="embed"` has no dataset validator.

<code-tabs name="val" />

## Export

<export-matrix />

Export bakes the model's current state into a fixed graph. For `task="classify"`, whatever labels `set_classes()` last set, and the resolution at export time, are baked into a final linear layer with the learned scale and bias, so the exported graph is an ordinary `[B, K]` image classifier with no text tower and no tokenizer; export again after changing the classes or the size. Exporting in `multi_label=True` mode is not implemented; set it back to `False` first. `task="embed"` export traces the image tower alone. Both need ONNX opset 14 or higher, which the exporter sets by default.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family. Both are converted from Google's Apache-2.0 `siglip2-base-patch16-256` and `siglip2-so400m-patch14-384` checkpoints, not from any COCO training run.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
