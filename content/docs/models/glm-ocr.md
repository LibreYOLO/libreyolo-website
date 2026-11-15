---
title: GLM-OCR
families: [glmocr]
seo_title: "GLM-OCR in Python: parse document pages to markdown"
description: "Run Z.ai's GLM-OCR in LibreYOLO: PP-DocLayoutV3 layout plus a 0.9B recognizer that reads text, tables (HTML) and formulas (LaTeX). MIT and Apache-2.0."
lead: "GLM-OCR is Z.ai's two-stage document parser. PP-DocLayoutV3 finds the layout regions and their reading order, and the 0.9B GLM-OCR recognizer reads each region as text, a table in HTML or a formula in LaTeX. LibreYOLO ports the official SDK pipeline around both networks for the ocr task."
keywords: [GLM-OCR, Z.ai, zai-org, document parsing, OCR to markdown, table recognition, formula recognition, PP-DocLayoutV3, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")            # weights download on first use
        result = model.predict("page.png")      # a scanned or rendered page

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])             # one block per region, reading order
        print(result.ocr.markdown)              # the whole page as markdown
    - label: One element
      language: python
      code: |
        from libreyolo import LibreVLM

        # Skip layout detection and read a pre-cropped table.
        model = LibreVLM("glm-ocr", use_layout_detection=False, prompt_label="table")
        result = model.predict("table.png")
        print(result.ocr.texts[0])              # HTML table
    - label: Extraction
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")
        reply = model.chat("id_card.png", 'Extract as JSON: {"name": "", "date": ""}')
        print(reply)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")
        metrics = model.val(data="ocr_dataset/")   # det hmean, end-to-end F1, 1-NED
---

## Install

GLM-OCR belongs to LibreYOLO's VLM tier and needs the `vlm` extra. Both
networks load through native transformers classes, which needs transformers
5.3 or newer.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights for both networks download on first use from one pinned mirror,
`LibreYOLO/LibreGLMOCR0.9b`. No remote code runs.

<code-tabs name="predict" />

The family loads through `LibreVLM()`, or the `LibreGLMOCR` class.
`Results.ocr` holds one block per layout region in reading order: `texts`,
`polygons` (axis-aligned boxes written as 4-point quads), `labels` (the layout
class, such as `text`, `table` or `formula`) and the page in `markdown`,
formatted as the official SDK does (titles, `\tag{}` formula numbers,
hyphenation and bullet fixes). Headers, footers, page numbers, footnotes and
side text are dropped, as upstream does; pictures and charts are not read.

`Results.ocr.conf` is the geometric mean of the generated tokens'
probabilities and `det_conf` the layout score. `conf=` does not gate regions;
the layout uses the upstream threshold 0.3, which `layout_threshold=` changes.
Decoding follows the SDK: greedy, repetition penalty 1.1, up to 8192 new
tokens. Each crop is round-tripped through JPEG before recognition as the SDK
sends it; `image_format=None` skips that.

`use_layout_detection=False` with `prompt_label=` (`text`, `table` or
`formula`) reads a pre-cropped element. It is a LibreYOLO convenience with no
SDK counterpart. `chat()` covers the JSON information-extraction prompts.

## Variants

One size, `0.9b`: a CogViT vision encoder and a GLM-0.5B decoder. The layout
stage runs at 800 px. CUDA runs the recognizer in bf16 (fp16 without bf16
support) and the layout detector in fp32; CPU and MPS run both in fp32. PDF
input, polygon crops, picture export and the hosted-API client are not ported.

## Validate

`val()` runs the shared OCR validator through the model's own `predict`.

<code-tabs name="val" />

## Export

Inference only: `train()` and `export()` raise.

## Licensing

<provenance-box>

The GLM-OCR recognizer weights are MIT, as declared by the
[Z.ai](https://github.com/zai-org/GLM-OCR) model card. The PP-DocLayoutV3
layout detector is Apache-2.0 (PaddlePaddle Authors), and the SDK pipeline
LibreYOLO ports is Apache-2.0. Commercial use is allowed. The weights are an
ungated mirror.

</provenance-box>

## Citation

<citation-block />
