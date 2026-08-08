---
title: PicoDet
families: [picodet]
seo_title: "PicoDet in LibreYOLO: predict, train and export"
description: "Run PicoDet in LibreYOLO for mobile object detection. Install, predict, train, validate and export under Apache-2.0."
lead: "PicoDet is a single-stage detector built for mobile and edge CPUs: an ESNet backbone, a CSP-PAN neck and a shared Generalized Focal Loss head. LibreYOLO supports it for detection."
keywords: [PicoDet, PP-PicoDet, object detection, mobile object detection, edge detection, ESNet, Generalized Focal Loss]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePICODETs.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            allow_experimental=True,
            epochs=300, batch=16, lr0=0.01,
        )
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320
        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibrePICODETs.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

PicoDet needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` sets the confidence threshold
and `iou` the NMS threshold. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Three sizes, each at its own fixed input resolution: `s` the smallest and
`l` the largest. Resolution grows with the size, so larger checkpoints are
also more expensive to run per image, on top of carrying more parameters.

<benchmark-table task="detect" />

<va-embed />

## Train

<code-tabs name="train" />

Training requires `allow_experimental=True`; leave it off and `train()`
raises, spelling out what has been checked and what has not. What is
validated: inference, plus ONNX, TorchScript, NCNN and OpenVINO export. What
is not: small-dataset fine-tune convergence, multi-GPU training, and any
augmentation beyond horizontal flip. The loss and assigner (VFL, DFL, GIoU,
SimOTA) match the upstream recipe and the trainer runs end to end, but the
smallest size's tiny backbone at its correspondingly small input resolution
is a poor fit for a handful of small, few-class fine-tunes; it is a better
fit at full-COCO scale.

`train()` also accepts a `pretrained` argument, but the value is never read
inside the method: training always continues from whatever weights the model
was constructed with, so `pretrained=False` does not reinitialize the
network. Leave `imgsz` unset and it takes the loaded checkpoint's native
resolution.

Left alone otherwise, the trainer runs 300 epochs with SGD at `lr0=0.01`,
momentum 0.9 and a 1-epoch warmup; horizontal flip is the only augmentation
applied.

The CLI has no `allow_experimental` flag, so `libreyolo train` cannot start a
PicoDet run; use the Python API shown above.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

LibreYOLO's port follows Bo396543018/Picodet_Pytorch, a PyTorch
re-implementation of PaddleDetection's original PP-PicoDet, with mmcv
stripped out and every activation matched exactly so PaddlePaddle checkpoints
converted through Bo's pipeline load with no numerical drift. Both sources
carry the same Apache-2.0 terms as the paper's authors.

</provenance-box>

## Citation

<citation-block />
