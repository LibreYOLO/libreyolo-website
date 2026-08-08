---
title: D-FINE
families: [dfine]
seo_title: "D-FINE: fine-tune, validate and export under MIT"
description: "Use D-FINE in LibreYOLO for object detection and instance segmentation. Install, predict, fine-tune, validate and export, with MIT-licensed code."
lead: "A detection transformer that reformulates box regression as a probability distribution over each box edge, refined across decoder layers. LibreYOLO supports it for detection and instance segmentation."
keywords: [D-FINE, detection transformer, real-time object detection, instance segmentation, fine-grained distribution refinement, DETR]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -seg suffix in the filename selects the mask head, so no task
        # argument is needed here.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Instance segmentation
      language: bash
      code: |
        # Continues from published segmentation weights, mask head included.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentation from detect weights
      language: bash
      code: |
        # Detect weights carry no mask head, so this is an explicit transfer:
        # the head starts untrained and is only useful once trained. Asking for
        # task=segment here is what authorizes the transfer.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masks
        print(metrics["metrics/mAP50-95(B)"])   # boxes
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640
        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

D-FINE needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

Adapter fine-tuning with `lora=True` is the exception, and needs the `lora`
extra.

```bash
pip install "libreyolo[lora]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. A `-seg` filename resolves to the
segmentation task on its own, and `result.masks` then carries the instance
masks alongside the boxes. `conf` and `max_det` filter the query selection;
`iou` is accepted for API parity but has no effect, because the decoder is a set
predictor with no NMS step. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Five sizes. They all run at the same input resolution, so the table separates
them by parameter count and accuracy.

<benchmark-table task="detect" />

<va-embed />

Segmentation reuses the detection backbone, encoder and decoder and adds a mask
head, so a `-seg` checkpoint takes the same arguments as its detect sibling.
LibreYOLO's RT-DETRv4 family is written as a subclass of the D-FINE wrapper: it
inherits this decoder line and then pins its task list back to detection,
because it carries no mask head.

## Train

Training starts from a published checkpoint, for both tasks.

<code-tabs name="train" />

Left alone, the trainer runs 132 epochs at `lr0=2e-4` with `amp=False`, a batch
of 16 and early stopping after 50 epochs without improvement. Detect weights are
a legal starting point for segmentation training, but only as an explicit
transfer, since the mask head begins untrained and would otherwise return
meaningless masks. Passing `task=segment` to the CLI is what authorizes it. The
Python route is narrower: `LibreDFINE` has to be constructed directly with
`allow_detect_to_segment_transfer=True`, because the `LibreYOLO()` factory takes
no such argument, and direct construction does not download, so the weights file
must already be on disk.

`lora=True` applies to detection. Segment training rejects it and points at
`freeze='backbone'` instead, because the mask head has not been tested with
adapters. On Apple silicon the trainer moves the whole run to CPU: the backward
pass of the Integral's binned matmul hits a Metal compilation failure. Inference
on MPS is unaffected.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary keyed by metric name, and prints per-class results
when `verbose` is left on.

<code-tabs name="val" />

Against a `-seg` checkpoint the plain `metrics/mAP50-95` key holds the mask
score, and the same run also reports boxes under `(B)` and masks under `(M)` so
both are available from one pass.

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. The OpenVINO, Paddle, MNN and Core AI paths export at a fixed canvas
rather than dynamic shapes. [Export](/docs/export) lists the arguments every
format accepts and the extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

The segmentation weights have a second upstream: their mask decoder, mask
matching and mask loss come from ArgoHA/D-FINE-seg, also Apache-2.0, whose
maintainer approved reuse with attribution.

</provenance-box>

## Citation

<citation-block />
