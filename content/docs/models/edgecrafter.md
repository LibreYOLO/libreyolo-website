---
title: EdgeCrafter
families: [ec]
seo_title: "EdgeCrafter: detect, pose and segment in LibreYOLO"
description: "Use EdgeCrafter in LibreYOLO for detection, pose and instance segmentation. Install, predict, validate and export, with MIT-licensed code."
lead: "A compact vision transformer for dense prediction on edge hardware, published upstream as three sibling models: ECDet, ECPose and ECSeg. LibreYOLO loads all three as one family, with the task carried by the checkpoint."
keywords: [EdgeCrafter, ECDet, ECPose, ECSeg, compact vision transformer, object detection, pose estimation, instance segmentation, edge inference]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs.pt source=bus.jpg save=True
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -pose suffix in the filename selects the keypoint head, so no
        # task argument is needed here.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50 imgsz=640 batch=8 lr0=5e-4
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Needs a single-class keypoint dataset whose data.yaml declares
        # kpt_shape, and imgsz at the checkpoint's native size.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Needs polygon labels, and imgsz at the checkpoint's native size.
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masks
        print(metrics["metrics/mAP50-95(B)"])   # boxes
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

EdgeCrafter needs no optional extra. Everything it imports is in the base
install.

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

The task comes from the filename, so a `-pose` or `-seg` checkpoint selects its
own head and takes no task argument. All three return the `Results` object every
family returns, with `result.keypoints` added for pose and
`result.masks` for segmentation. Pose covers one class, person, with the 17
COCO keypoints, and the count is fixed when the model is built. It has no box
head, so each pose box is the bounding extent of its own keypoints, and the
third keypoint channel is a constant rather than a per-point score.

`conf` and `max_det` filter the query selection; `iou` is accepted for API
parity but has no effect, because all three heads decode a set of queries with no
NMS step. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Variants

Four sizes. They all run at the same input resolution, so the table separates
them by parameter count and accuracy.

<benchmark-table task="detect" />

<va-embed />

Upstream publishes ECDet, ECPose and ECSeg as three separate models rather than
one model with three heads. They share the ECViT backbone and the hybrid encoder
and differ only in the head, so LibreYOLO folds them into a single family and
lets the checkpoint filename carry the task. A size letter therefore means the
same backbone and encoder across all three, and predict, validate and export
take the same arguments whichever one you load.

## Train

All three tasks train through `train()`, which reads the task from the loaded
checkpoint and picks the matching trainer.

<code-tabs name="train" />

What has been checked for detection and segmentation: inference parity against
upstream at 1e-5, layer by layer and per size, and that the loss and a single
training step run on synthetic input. What has not, per `train()`'s own
docstring: convergence of a full fine-tune, multi-GPU training, the
stop-augmentation best-reload step, and the Objects365 to COCO class remap. The
pose path follows DETRPose's published recipe, a Hungarian matcher over class,
keypoint L1 and OKS costs with contrastive keypoint denoising, and its
convergence has not been checked end to end either.

Left alone, the trainer runs 74 epochs at `lr0=5e-4` with mixed precision on,
following upstream's recipe: AdamW, a flat cosine schedule, EMA at 0.9999 and
ImageNet-normalized inputs. Pose and segmentation both require `imgsz` at the
checkpoint's native size, because their evaluation anchor grid is built when the
model is constructed; a different value raises before the run starts. Pose also
requires a single-class dataset whose `data.yaml` declares `kpt_shape`, with a
keypoint count matching the head.

`lora=True` applies to detection only; pose and segmentation raise a
`ValueError` on it. On Apple silicon the trainer keeps the run on the GPU and
sends one operation to CPU, the grid-sample backward inside deformable
attention, which PyTorch does not implement in Metal.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary keyed by metric name, and prints per-class results
when `verbose` is left on.

<code-tabs name="val" />

Pose reports keypoint OKS metrics under `metrics/keypoints_*`. Segmentation
reports masks under the plain `metrics/mAP50-95` key and repeats both views in
one pass, boxes under `(B)` and masks under `(M)`.

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Pose and segmentation export at a fixed 640 by 640 input rather than
dynamic shapes, and several detection targets are fixed-canvas too, including
OpenVINO, Paddle, MNN, ExecuTorch and Core AI. [Export](/docs/export) lists the
arguments every format accepts and the extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
