---
title: FOMO
families: [fomo]
seo_title: "FOMO: point localization, train and export in LibreYOLO"
description: "Run FOMO (Faster Objects, More Objects) in LibreYOLO: a tiny point-localization detector for counting many small objects. Install, predict, train and export."
lead: "FOMO is a grid-based point localizer: each cell of a low-resolution grid is classified as background or an object center, with no bounding-box regression. LibreYOLO supports it for the point task."
keywords: [FOMO, Faster Objects More Objects, point localization, centroid detection, tiny object detection, edge AI, MCU detection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMO weights are not auto-downloaded (see Checkpoints below).
        # Point this at a checkpoint you already downloaded locally.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=./LibreFOMOs-point.pt source=bus.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: |
        # imgsz must be passed: the CLI defaults it to 640, and the s
        # checkpoint accepts only its native 96.
        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
---

## Install

FOMO needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Unlike every other family on this site, LibreFOMO weights are not
auto-downloaded: `LibreYOLO("LibreFOMOs-point.pt")` looks for that file on
disk and raises a `ValueError` naming it rather than fetching it from Hugging
Face. Download a checkpoint from the [LibreYOLO org](https://huggingface.co/LibreYOLO)
first and load it by local path, or train your own (see Train below).

<code-tabs name="predict" />

The result carries a `points` payload instead of `boxes`: each row is
`x, y, class, confidence`, available as `result.points.data`, or through
the `.xy`, `.xyn`, `.cls` and `.conf` accessors. There is no `iou` threshold
to set, because there are no boxes to suppress; `predict(..., nms_radius=1)`
controls how many grid cells apart two detections must be to both survive,
and the filename must carry FOMO's `-point` task suffix for the loader to
recognize it. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

Three sizes, `s`, `m` and `l`, use progressively wider MobileNetV2-style
backbones at correspondingly larger, fixed input resolutions, each behind a
single 1x1 classification head. This family carries no benchmark table here;
checkpoint file size in the table below is the clearest per-size signal
currently published.

## Train

<code-tabs name="train" />

`imgsz` is not a free choice: it defaults to the loaded checkpoint's native
resolution, and passing a different value raises `ValueError` naming the size
it expects. Those sizes are 96 for `s`, 192 for `m` and 224 for `l`. The CLI
defaults `imgsz` to 640, so a `libreyolo train` command has to set it
explicitly to match the checkpoint.

Left alone otherwise, the trainer runs 40 epochs at batch 32 with Adam at
`lr0=3e-4`, no weight decay, and a foreground class weighted 100x over
background in the per-cell cross-entropy loss, since almost every grid cell is
background in a typical scene. EMA and mixed precision are both off by
default, and none of the geometric or color augmentations used elsewhere in
LibreYOLO are applied: mosaic, mixup, HSV jitter, flip, rotation, translation
and shear are all zero.

This is the path the published LibreFOMO checkpoints were trained with, from
scratch on COCO.

See [training](/docs/train) for datasets and loggers.

## Validate

`val()` dispatches to a grid-level validator built for this family. Alongside
the point-matching `metrics/precision`, `metrics/recall` and `metrics/mAP@`
keys shared with other point tasks, it sweeps confidence thresholds and
`nms_radius` values and publishes the best-F1 combination under
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall` and
`metrics/grid_mean_distance`, plus the threshold and radius that produced it
under `decode/threshold` and `decode/nms_radius`.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family. None of them download
automatically: fetch the file you want from the linked Hugging Face page and
pass its local path to `LibreYOLO()`.

<checkpoint-table />

## Licensing

<provenance-box>

There is no upstream code repository for FOMO to link: Edge Impulse describes
the technique through a blog post and its product documentation, but has not
released FOMO training or inference code. The architecture and training here
are LibreYOLO's own implementation of that published description, and the
published LibreFOMO checkpoints are trained from scratch on COCO, so both the
code and these weights are MIT, LibreYOLO's own. The name FOMO and the
technique it describes remain Edge Impulse's.

</provenance-box>

## Citation

Edge Impulse announced FOMO in a blog post rather than a paper, and publishes
no BibTeX block for it. Cite the announcement linked in the Upstream row
above.
