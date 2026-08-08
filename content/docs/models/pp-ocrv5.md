---
title: PP-OCRv5
families: [ppocr]
seo_title: "PP-OCRv5: text detection and recognition in LibreYOLO"
description: "Use PP-OCRv5 in LibreYOLO for multilingual scene-text OCR. Install, predict and validate the t and l checkpoints, Apache-2.0 licensed."
lead: "PP-OCRv5 is PaddleOCR's text detection and recognition pipeline: a differentiable-binarization detector locates text quads and an SVTR/CTC recognizer reads them. LibreYOLO ports it to PyTorch for two tiers."
keywords: [PP-OCRv5, PaddleOCR, OCR, text detection, text recognition, scene text]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Quads
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # (N, 4, 2) polygons in reading order: top-left, top-right,
        # bottom-right, bottom-left. Detection quads are genuine polygons
        # (rotated text), so they populate result.ocr, not result.boxes.
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # headline metric
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
---

## Install

PP-OCRv5 needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

Each checkpoint bundles both stages, detection and recognition, under one
`.pt` file, with the recognition charset and pipeline defaults carried in the
checkpoint metadata. The recognizer reads Simplified and Traditional Chinese,
English, Japanese and pinyin with one dictionary. `result.ocr` is an
`OCRRegions` payload: `.data` holds the four-point polygons, `.texts` the
transcripts, `.conf` the per-region recognition score, and
`.det_conf` the detection score. Multi-image sources run sequentially:
the two-stage pipeline does not batch across images. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two tiers: `t`, built on lighter PP-LCNetV3/PP-OCRv5_mobile backbones for CPU
use, and `l`, built on PP-HGNetV2 server backbones for higher accuracy. Both
tiers run detection at a fixed long-side limit and recognize crops in batches;
`rec_batch` controls how many crops go through the recognizer per forward
pass.

## Validate

`val()` measures the pipeline against a directory of images plus a
`labels/<split>.jsonl` file, or the equivalent dataset YAML, each label
listing per-image text-region polygons and their transcripts. It reports
detection hmean (IoU-matched precision/recall/F1), end-to-end F1 (hmean plus an
exact transcript match after normalization, the checkpoint's fitness metric),
and 1-NED, the mean normalized edit distance over matched pairs.

<code-tabs name="val" />

## Export

<export-matrix />

PP-OCRv5 is a two-network pipeline, detection and recognition moving
together, not one traceable graph, and export is not implemented for it: no
format is supported yet. Fine-tune the Apache-2.0 upstream training code
directly and convert the result with `weights/convert_ppocr_weights.py` if you
need a checkpoint outside this format.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
