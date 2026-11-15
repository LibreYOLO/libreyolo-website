---
title: RiO-DETRv2
families: [riodetrv2]
seo_title: "RiO-DETRv2 in Python: real-time oriented object detection, Apache-2.0"
description: "Use RiO-DETRv2 in LibreYOLO for oriented-box detection on aerial images. Sizes n to x, DOTA-v1.0 and DIOR-R checkpoints, Apache-2.0 code and weights."
lead: "RiO-DETRv2 is a real-time oriented-box DETR. It predicts four box sides as distributions in the box's own frame, refines the angle coarse to fine and scores each box with an angle-aware quality. LibreYOLO ports it inference-only, in five sizes, with DOTA-v1.0 and DIOR-R checkpoints."
keywords: [RiO-DETRv2, RiO-DETR, oriented object detection, OBB, rotated boxes, DOTA, DIOR-R, aerial images, DETR]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb.pt")        # DOTA-v1.0, 15 classes
        result = model("aerial.jpg", save=True)

        print(result.obb.xywhr)   # (N, 5) centre, size and angle
        print(result.obb.conf, result.obb.cls)
    - label: DIOR-R
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb-diorr.pt")  # DIOR-R, 20 classes
        result = model("aerial.jpg")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRiODETRv2s-obb.pt source=aerial.jpg save=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb.pt")
        model.export(format="onnx")
        model.export(format="torchscript")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRiODETRv2s-obb.pt format=onnx
---

## Install

RiO-DETRv2 needs no optional extra.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face mirrors on first use. The
models are trained on aerial imagery, so use an overhead photo.

<code-tabs name="predict" />

`result.obb` holds the rotated boxes: `xywhr` (centre, width, height and
angle), `conf` and `cls`, with class names from the checkpoint. See the
[oriented detection](/docs/tasks/oriented-detection) task page for the
payload and plotting.

## Variants

Five sizes, `n`, `s`, `m`, `l` and `x`, each with two checkpoints:

| Checkpoint | Dataset | Classes | Canvas |
| --- | --- | ---: | ---: |
| `LibreRiODETRv2<size>-obb.pt` | DOTA-v1.0 | 15 | 1024 px |
| `LibreRiODETRv2<size>-obb-diorr.pt` | DIOR-R | 20 | 800 px |

The native canvas is read from the checkpoint. Raw outputs and decoded
rotated boxes match the upstream Apache-2.0 release exactly (max absolute
difference 0) for all ten checkpoints on CPU.

RiO-DETRv2 is inference-only. `train()` raises: the upstream recipe (DINOv3
feature distillation, OBB Hungarian matching, boundary and angle
distribution losses) is not ported. For trainable oriented-box detection use
[GTR](/docs/models/gtr) or [YOLO-NAS](/docs/models/yolo-nas).

## Export

<export-matrix />

ONNX and TorchScript exports round-trip: the exported file loads back through
`LibreYOLO()` and returns the same `Results`.

<code-tabs name="export" />

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box>

RiO-DETRv2 is by RicePasteM
([RicePasteM/RiO-DETRv2](https://github.com/RicePasteM/RiO-DETRv2)). Code and
weights are Apache-2.0, so commercial use is fine. Upstream changed its
LICENSE file to AGPL-3.0 about 33 minutes after its initial commit, with no
code changes. LibreYOLO's port uses only that initial Apache-2.0 release
(commit `ab8b22d`) and preserves the snapshot at `LibreYOLO/RiO-DETRv2`. The
official checkpoints are mirrored as `LibreYOLO/LibreRiODETRv2{n,s,m,l,x}-obb`
and `-obb-diorr`, with keys renamed and learned tensors unchanged. The
DOTA-v1.0 and DIOR-R training sets carry their own terms.

</provenance-box>

## Citation

<citation-block />
