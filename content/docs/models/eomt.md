---
title: EoMT
families: [eomt]
seo_title: "EoMT: predict semantic, instance and panoptic segmentation"
description: "Use EoMT in LibreYOLO for semantic, instance and panoptic segmentation on a plain DINOv2 vision transformer, no decoder needed. MIT-licensed."
lead: "A segmentation network built on a plain vision transformer with no dedicated pixel decoder: extra learned queries added to the encoder itself predict the masks. LibreYOLO supports it for semantic, instance and panoptic segmentation."
keywords: [EoMT, encoder-only mask transformer, DINOv2, panoptic segmentation, instance segmentation, semantic segmentation]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Semantic
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        results = model(SAMPLE_IMAGE, save=True)

        mask = results[0].semantic_mask
        print(mask.data.shape)   # (H, W) class ids
        print(mask.classes)      # sorted class ids present in the image
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -seg suffix in the filename selects the instance task, so no
        # task argument is needed here.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        results = model(SAMPLE_IMAGE, save=True)

        print(results[0].boxes.xyxy)
        print(results[0].masks.data.shape)
    - label: Panoptic
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        results = model(SAMPLE_IMAGE, save=True)

        pan = results[0].panoptic
        print(pan.data.shape)       # (H, W) segment ids
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-sem.pt source=bus.jpg save=True
  val:
    - label: Semantic
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masks
        print(metrics["metrics/mAP50-95(B)"])   # boxes
    - label: Panoptic
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].semantic_mask.data.shape)
---

## Install

EoMT needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally. The
task suffix in the filename (`-sem`, `-seg`, `-panoptic`) selects the task, and
`LibreYOLO()` infers it from that filename so no `task=` argument is needed.

<code-tabs name="predict" />

Semantic segmentation fills `results[0].semantic_mask`, a `(H, W)` array of
class ids on `.data`. Instance segmentation fills `results[0].boxes` and
`results[0].masks`, the same shape every other segmentation family returns.
Panoptic segmentation fills `results[0].panoptic`: a `(H, W)` segment-id map on
`.data`, plus `.segments_info`, a list of `{"id", "category_id"}` dicts, one
per segment. `conf` filters query selection; `iou` has no effect on the
semantic task, since it argmaxes per pixel with no NMS step. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Three encoder sizes, s/b/l, all DINOv2-backed. The semantic checkpoint is
trained on ADE20K at 512 px; the instance and panoptic checkpoints are trained
on COCO at 640 px, with a second instance checkpoint trained at 1280 px.
Upstream ships DINOv2 instance-segmentation weights only at size l; s and b are
published for semantic and panoptic only. DINOv3-backed EoMT variants exist
upstream but are not shipped here, because they depend on gated non-commercial
DINOv3 weights.

LibreYOLO does not train EoMT: `train()` raises `NotImplementedError` for this
family, which the [support tier](/docs/models) above marks as inference only.

## Validate

`val()` dispatches by task. Semantic returns `metrics/mIoU` and
`metrics/pixel_accuracy`. Instance segmentation returns the same mask and box
mAP keys as other segmentation families. Panoptic returns Panoptic Quality as
`metrics/PQ`, split into `metrics/SQ` (segmentation quality) and `metrics/RQ`
(recognition quality), plus `metrics/PQ_things` and `metrics/PQ_stuff`.

<code-tabs name="val" />

## Export

<export-matrix />

Only the semantic task exports today: instance and panoptic segmentation call
`export()` and get `NotImplementedError`, because their query-mask output has
no runtime export contract yet. An exported semantic artifact loads back
through `LibreYOLO()` on its file suffix, so a `.onnx` or `.engine` file
behaves like a checkpoint and returns the same `Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
