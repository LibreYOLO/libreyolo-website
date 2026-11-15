---
title: PP-DocLayoutV3
families: [ppdoclayoutv3]
seo_title: "PP-DocLayoutV3 in Python: document layout detection with reading order"
description: "Use PP-DocLayoutV3 in LibreYOLO to find 25 kinds of page regions and rank them in reading order. Apache-2.0, pure PyTorch, ONNX and TorchScript export."
lead: "PP-DocLayoutV3 is PaddleOCR's document layout detector. It finds the regions of a page (text, titles, tables, figures, formulas, headers, footers, seals and more) and ranks them in reading order. LibreYOLO ports it as a detect family that needs no transformers or PaddlePaddle runtime."
keywords: [PP-DocLayoutV3, PaddleOCR, document layout analysis, layout detection, reading order, RT-DETR, HGNetV2, document parsing]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.pt")
        result = model("page.png", conf=0.5)    # the upstream pipeline's threshold

        for order, box, cls, score in zip(
            result.reading_order, result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            print(int(order), result.names[int(cls)], round(float(score), 3), box.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPDocLayoutV3l.pt source=page.png conf=0.5 save=True
    - label: Official weights
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Upstream's model.safetensors loads too: it is converted once and a
        # LibreYOLO checkpoint is written beside it.
        model = LibreYOLO("model.safetensors")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.pt")
        model.export(format="onnx")             # fixed 800x800, FP32
        model.export(format="torchscript")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePPDocLayoutV3l.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.onnx")
        result = model("page.png", conf=0.5)
        print(result.reading_order)
---

## Install

PP-DocLayoutV3 needs no optional extra. It is pure PyTorch and everything it
imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from `LibreYOLO/LibrePPDocLayoutV3l` on first use.

<code-tabs name="predict" />

`Results.boxes` holds xyxy boxes on the original page, confidences and class
ids over 25 layout classes. Rows are sorted in reading order, and
`Results.reading_order` is a one-based int64 vector aligned with the boxes. It
slices with `results[i]` like `track_id` and appears as `reading_order` in
`Results.summary()` rows. Ranks are computed among the regions kept at `conf`;
a `classes=` filter or a `max_det` cut keeps the surviving ranks, so the values
can have gaps.

The upstream pipeline uses `conf=0.5`; LibreYOLO's default `conf=0.25` keeps
more low-confidence regions. As upstream, scores are per-class sigmoids and the
top 300 (query, class) pairs are kept, so one region can be reported with two
labels. No NMS is applied.

Classes: `abstract`, `algorithm`, `aside_text`, `chart`, `content`,
`display_formula`, `doc_title`, `figure_title`, `footer`, `footer_image`,
`footnote`, `formula_number`, `header`, `header_image`, `image`,
`inline_formula`, `number`, `paragraph_title`, `reference`,
`reference_content`, `seal`, `table`, `text`, `vertical_text`,
`vision_footnote`.

The same port runs the layout stage of [Falcon OCR](/docs/models/falcon-ocr)
with `layout=True`.

## Variants

One size, `l`: an RT-DETR-style set-prediction detector with an HGNetV2-L
backbone, mask-enhanced query initialisation and a global-pointer head that
scores every pair of regions. Input is a fixed 800x800 canvas (stretch resize,
RGB in `[0, 1]`, no normalization). It runs on CPU, CUDA and MPS; on MPS the
positional encoding is computed in float32, so results can differ in the last
bits.

Against the Apache-2.0 PyTorch reference in transformers, on CUDA in fp32 with
the same input tensor, logits, boxes, reading-order logits and masks match
exactly on three real pages and a random tensor. Training is not offered: it
needs reading-order and per-region mask supervision that LibreYOLO's detect
datasets do not carry. The per-region polygons upstream derives from the mask
branch are not exposed.

## Export

<export-matrix />

ONNX (opset 17) and TorchScript, fixed 800x800, FP32. The graph outputs
`pred_logits`, `pred_boxes` and `order_votes`; exported backends return the
same `Results`, including `reading_order`.

<code-tabs name="export" />

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box>

PP-DocLayoutV3 is Apache-2.0, code and weights, by the
[PaddlePaddle Authors](https://github.com/PaddlePaddle/PaddleOCR). Commercial
use is allowed. The checkpoint is converted from
`PaddlePaddle/PP-DocLayoutV3_safetensors` by stripping a key prefix and adding
metadata; learned parameters are unchanged and the mirror is not gated.

</provenance-box>

## Citation

<citation-block />
