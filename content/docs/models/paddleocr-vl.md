---
title: PaddleOCR-VL
families: [paddleocrvl]
seo_title: "PaddleOCR-VL in Python: document parsing with a 0.9B VLM"
description: "Run PaddleOCR-VL in LibreYOLO: PP-DocLayoutV2 layout plus a 0.9B recognizer for text, tables (HTML), formulas (LaTeX) and charts. Apache-2.0."
lead: "PaddleOCR-VL is PaddlePaddle's two-stage document parser. PP-DocLayoutV2 finds 25 kinds of layout region and their reading order, and PaddleOCR-VL-0.9B reads each region as text, a table, a formula or a chart. LibreYOLO ports the PaddleX pipeline around both networks for the ocr task."
keywords: [PaddleOCR-VL, PaddleOCR, PaddlePaddle, document parsing, OCR to markdown, table recognition, formula recognition, ERNIE, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("paddleocr-vl")       # weights download on first use
        result = model.predict("page.png")      # a scanned or rendered page

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])             # one region per block, reading order
        print(result.ocr.markdown)              # the whole page as markdown
    - label: One element
      language: python
      code: |
        from libreyolo import LibreVLM

        # Skip layout detection and read a pre-cropped formula.
        model = LibreVLM("paddleocr-vl", use_layout_detection=False, prompt_label="formula")
        result = model.predict("formula.png")
        print(result.ocr.texts[0])              # LaTeX
    - label: Charts
      language: python
      code: |
        from libreyolo import LibreVLM

        # Charts are skipped by default, as upstream; opt in to read them as tables.
        model = LibreVLM("paddleocr-vl", use_chart_recognition=True)
        result = model.predict("report.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("paddleocr-vl")
        metrics = model.val(data="ocr_dataset/")   # det hmean, end-to-end F1, 1-NED
---

## Install

PaddleOCR-VL belongs to LibreYOLO's VLM tier and needs the `vlm` extra. Both
networks load through native transformers classes, which needs transformers
5.3 or newer. PaddlePaddle itself is not needed.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights for both networks download on first use from one pinned mirror,
`LibreYOLO/LibrePaddleOCRVL0.9b`. No remote code runs.

<code-tabs name="predict" />

The family loads through `LibreVLM()`, or the `LibrePaddleOCRVL` class.
`Results.ocr` holds one region per recognized block (a paragraph, a title, a
table), not one per text line, in reading order: `texts`, `polygons`
(axis-aligned boxes as 4-point quads), `labels` (the layout class) and the
page in `markdown`. Tables come back as HTML and formulas as LaTeX. Pictures,
and charts and seals by default, are left out of `Results.ocr` and the
markdown.

`Results.ocr.conf` is the geometric mean of the generated tokens'
probabilities and `det_conf` the layout score. `conf=` does not gate regions;
the layout keeps the upstream per-class thresholds, which `layout_threshold=`
overrides. Recognition is greedy with up to 4096 new tokens, and each region is
resized within `min_pixels` / `max_pixels` (default 112896 / 1003520).

`use_layout_detection=False` with `prompt_label=` (`ocr`, `table`, `formula`
or `chart`) reads a pre-cropped element.

## Variants

One size, `0.9b`: a NaViT-style vision encoder and the ERNIE-4.5-0.3B decoder,
behind PP-DocLayoutV2. CUDA runs the recognizer in bf16 (fp16 without bf16
support) and the layout detector in fp32; CPU and MPS run both in fp32. On MPS,
set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an operator is reported missing.

Not ported: adjacent text-block merging, figures embedded in tables, seal and
spotting recognition, polygon layout shapes, document unwarping and image
export. The markdown is the plain variant, with no HTML centering.

## Validate

`val()` runs the shared OCR validator.

<code-tabs name="val" />

## Export

Inference only: `train()` and `export()` raise.

## Licensing

<provenance-box>

PaddleOCR-VL-0.9B and PP-DocLayoutV2 are Apache-2.0, code and weights, by the
[PaddlePaddle Authors](https://github.com/PaddlePaddle/PaddleOCR). Commercial
use is allowed. The pipeline LibreYOLO ports from PaddleX is also Apache-2.0.
The weights are an unmodified, ungated mirror.

</provenance-box>

## Citation

<citation-block />
