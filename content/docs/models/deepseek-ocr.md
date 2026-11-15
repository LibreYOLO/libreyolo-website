---
title: DeepSeek-OCR
families: [deepseekocr]
seo_title: "DeepSeek-OCR and DeepSeek-OCR-2 in Python: document OCR to markdown"
description: "Run DeepSeek-OCR and DeepSeek-OCR-2 in LibreYOLO: text lines with boxes, or the whole page as markdown. MIT and Apache-2.0, no remote code."
lead: "DeepSeek-OCR is DeepSeek's 3B mixture-of-experts model for documents. LibreYOLO runs DeepSeek-OCR and DeepSeek-OCR-2 on the ocr task through a vendored PyTorch port: text lines with boxes, layout blocks, or the whole page as markdown."
keywords: [DeepSeek-OCR, DeepSeek-OCR-2, document OCR, OCR to markdown, vision-language model, mixture of experts, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr")      # weights download on first use
        result = model.predict("page.png")     # a scanned or rendered page

        for text, quad in zip(result.ocr.texts, result.ocr.polygons):
            print(text, quad.tolist())         # one region per text line
    - label: Markdown
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr", prompt="markdown")
        result = model.predict("page.png")

        print(result.ocr.markdown)             # the whole page as markdown
        print(result.ocr.labels)               # layout category per block
    - label: DeepSeek-OCR-2
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr-2")     # resolution "gundam" or "base"
        result = model.predict("page.png")
        print(result.ocr.texts)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr")
        metrics = model.val(data="ocr_dataset/")   # det hmean, end-to-end F1, 1-NED
---

## Install

DeepSeek-OCR belongs to LibreYOLO's VLM tier and needs the `vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download on first use from byte-identical LibreYOLO mirrors,
`LibreYOLO/LibreDeepSeekOCR1` and `LibreYOLO/LibreDeepSeekOCR2`, pinned to a
revision. The architecture is a PyTorch port of the release's inference path;
no remote code runs.

<code-tabs name="predict" />

The family loads through `LibreVLM()`, or the `LibreDeepSeekOCR` class. The
`prompt` setting (constructor or `set_prompt()`) picks what lands in
`Results.ocr`:

| `prompt` | `Results.ocr` |
| --- | --- |
| `"ocr"` (default) | one region per text line, `labels` all `text`, `markdown` empty |
| `"markdown"` | one region per layout block, the layout category in `labels`, the page in `markdown` |
| `"free"` | no regions, the page text in `markdown` |
| any other string | whatever grounding the model emits, plus `markdown` |

Boxes come on the model's 0 to 999 grid and are scaled to the original image
as axis-aligned 4-point quads, sorted top to bottom then left to right. A
generative model emits no score, so `conf` and `det_conf` are a constant 1.0.
`model.chat("page.png", "Parse the figure.")` returns raw generated text for
any prompt. Decoding is greedy, up to 8192 new tokens, with
`no_repeat_ngram_size=20` as in the release.

## Variants

| Name | Model | Resolutions | License |
| --- | --- | --- | --- |
| `deepseek-ocr` | DeepSeek-OCR (size `1`) | `tiny` (512), `small` (640), `base` (1024), `large` (1280), `gundam` (default) | MIT |
| `deepseek-ocr-2` | DeepSeek-OCR-2 (size `2`) | `base` (1024), `gundam` (default) | Apache-2.0 |

`gundam` combines a 1024 px global view with local crops: 2 to 9 crops of
640 px on DeepSeek-OCR, up to 6 crops of 768 px on DeepSeek-OCR-2. DeepSeek-OCR-2
keeps the decoder and the SAM stage and replaces CLIP with a Qwen2 encoder;
prompts and outputs are the same.

CUDA runs the bf16 weights under `torch.autocast`, as the release does (fp16
on GPUs without bf16). CPU and MPS run float32 with eager attention, about
13 GB of memory; MPS has not been run on hardware. EXIF orientation is not
applied. On an RTX 3090 the port reproduces the release token for token on
17 cases for DeepSeek-OCR and 14 for DeepSeek-OCR-2. Benchmark numbers are in
the release papers and are not reproduced by LibreYOLO.

## Validate

`val()` runs the shared OCR validator and reports detection hmean,
end-to-end F1 and 1-NED.

<code-tabs name="val" />

## Export

Inference only: `train()` and `export()` raise.

## Licensing

<provenance-box>

DeepSeek-OCR is MIT and DeepSeek-OCR-2 is Apache-2.0, code and weights, both
by [DeepSeek](https://github.com/deepseek-ai/DeepSeek-OCR). Both allow
commercial use. The weights are byte-identical, ungated mirrors.

</provenance-box>

## Citation

<citation-block />
