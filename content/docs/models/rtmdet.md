---
title: RTMDet
families: [rtmdet]
seo_title: "RTMDet in LibreYOLO: predict, train and export"
description: "Run RTMDet in LibreYOLO for object detection and RTMDet-Ins instance segmentation. Install, predict, train, validate and export under Apache-2.0."
lead: "RTMDet is a single-stage detector that predicts from one point-based prior per grid location, no anchors, through a head whose convolutions are shared across feature levels. LibreYOLO supports it for detection and RTMDet-Ins instance segmentation."
keywords: [RTMDet, object detection, instance segmentation, RTMDet-Ins, anchor-free detection, mmdetection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTMDets.pt source=bus.jpg save=True
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -seg suffix in the filename selects the RTMDet-Ins mask head,
        # so no task argument is needed here.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        results = model(SAMPLE_IMAGE, save=True)

        print(results[0].masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masks
        print(metrics["metrics/mAP50-95(B)"])   # boxes
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            allow_experimental=True,
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640
        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRTMDets.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

RTMDet needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. A `-seg` filename resolves to the
RTMDet-Ins task on its own, and `results[0].masks` then carries the instance
masks alongside the boxes. `conf` sets the confidence threshold and `iou` the
NMS threshold. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

Five sizes, `t` through `x`, share one architecture at a common input
resolution. This family carries no benchmark table here: compare sizes by
checkpoint file size in the table below.

## Train

<code-tabs name="train" />

Detection training requires `allow_experimental=True`; leave it off and
`train()` raises, spelling out what has been checked and what has not. What is
validated: forward and ONNX export are bit-equivalent to upstream mmdetection,
and postprocessing matches mmdet's output within 0.001 mAP on val2017 subsets.
What is not: small-dataset fine-tune convergence, from-scratch paper parity,
multi-GPU behavior, cached Mosaic/MixUp throughput, and the strict upstream
two-stage pipeline switch. RTMDet-Ins (`task="segment"`, a `-seg` checkpoint)
has no training path at all yet: `train()` raises `NotImplementedError`
regardless of `allow_experimental`, before that flag is even checked.
Instance segmentation currently supports inference and validation only.

`train()` also accepts a `pretrained` argument, but the value is never read
inside the method: training always continues from whatever weights the model
was constructed with, so `pretrained=False` does not reinitialize the
network.

Left alone otherwise, the trainer runs 300 epochs with AdamW at `lr0=0.004`
and `weight_decay=0.05`, a 1-epoch warmup, and Mosaic and MixUp switched off
for the final 20 epochs.

The CLI has no `allow_experimental` flag, so `libreyolo train` cannot start an
RTMDet run; use the Python API shown above.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

Against a `-seg` checkpoint the plain `metrics/mAP50-95` key holds the mask
score, and the same run also reports boxes under `(B)` and masks under `(M)`
so both are available from one pass.

## Export

<export-matrix />

Detection exports to most formats; instance segmentation currently exports to
none of them; the matrix above reflects that split. An exported detection
artifact loads back through `LibreYOLO()` on its file suffix, so a `.onnx` or
`.engine` file behaves like a checkpoint and returns the same `Results`.
Running the graph in a bare runtime, with no LibreYOLO installed, is also
supported, but then preprocessing and postprocessing are yours to write.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
