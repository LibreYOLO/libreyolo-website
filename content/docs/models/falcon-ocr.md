---
title: Falcon OCR
families: [falconocr]
seo_title: "Falcon OCR in Python: a 0.27B document OCR model, Apache-2.0"
description: "Run TII's Falcon OCR in LibreYOLO: a whole page to text in one pass, or region by region with PP-DocLayoutV3. Apache-2.0, runs on CPU."
lead: "Falcon OCR is a 0.27B early-fusion document OCR model by the Technology Innovation Institute (TII). One transformer reads image patches and text together and writes the page as text, with LaTeX for formulas and HTML for tables. LibreYOLO runs a PyTorch port of Falcon OCR 1.5 on the ocr task."
keywords: [Falcon OCR, TII, document OCR, early fusion, small OCR model, OCR to markdown, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr")         # weights download on first use
        result = model.predict("page.png")      # a scanned or rendered page

        print(result.ocr.markdown)              # the page text
    - label: With layout
      language: python
      code: |
        from libreyolo import LibreVLM

        # PP-DocLayoutV3 finds regions in reading order; each crop is read
        # with its category prompt.
        model = LibreVLM("falcon-ocr", layout=True)
        result = model.predict("page.png")

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])
    - label: One element
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr", prompt="table")
        result = model.predict("table.png")     # a cropped table
        print(result.ocr.markdown)              # HTML table
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr")
        metrics = model.val(data="ocr_dataset/")
---

## Install

Falcon OCR belongs to LibreYOLO's VLM tier and needs the `vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download on first use from `LibreYOLO/LibreFalconOCR270m`, a
byte-identical mirror of `tiiuae/Falcon-OCR` pinned to a revision. The release
runs on FlexAttention and a Triton kernel through remote code; LibreYOLO runs a
plain PyTorch port with neither.

<code-tabs name="predict" />

The family loads through `LibreVLM()`, or the `LibreFalconOCR` class. There
are two modes, as upstream:

- Default: the whole page is read in one pass. `Results.ocr` holds one region
  covering the image, label `plain`, and `Results.ocr.markdown` is the same
  text.
- `layout=True`: LibreYOLO's [PP-DocLayoutV3](/docs/models/pp-doclayoutv3) port
  finds regions in reading order at `layout_threshold=0.3`, nested boxes are
  dropped, and each crop is read with its category prompt. One region per crop,
  `labels` holds the layout class, `det_conf` the layout score, and `markdown`
  joins the texts with blank lines. Pictures and classes without an upstream
  prompt category are skipped.

`prompt=` takes a category (`plain`, `text`, `table`, `formula`, `caption`,
`footnote`, `list-item`, `page-footer`, `page-header`, `section-header`,
`title`) or a custom instruction. Pages are resized so both sides fall between
64 and 1536 px. Decoding is greedy with a budget of 8192 new tokens. A
generative model emits no score, so `conf` is a constant 1.0.

## Variants

One size, `270m`. CUDA runs the bf16 weights under `torch.autocast`; CPU and
MPS run float32, about 1.1 GB of weights. Batches of one. On CPU in float32 the
port reproduces the release token for token on six OCR fixtures and a
synthetic report page; CUDA bf16 and the layout mode were not compared against
upstream.

## Validate

`val()` runs the shared OCR validator.

<code-tabs name="val" />

## Export

Inference only: `train()` and `export()` raise.

## Licensing

<provenance-box>

Falcon OCR is Apache-2.0, code and weights, by the
[Technology Innovation Institute](https://github.com/tiiuae/Falcon-Perception).
Commercial use is allowed. The weights are an ungated, byte-identical mirror.
The optional layout stage uses LibreYOLO's Apache-2.0 PP-DocLayoutV3 port.

</provenance-box>

## Citation

<citation-block />
