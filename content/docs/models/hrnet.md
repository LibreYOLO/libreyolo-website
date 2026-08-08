---
title: HRNet
families: [hrnet]
seo_title: "HRNet: top-down pose estimation in LibreYOLO"
description: "Use HRNet in LibreYOLO for top-down COCO-17 pose estimation. Install, predict, validate and export the W32 and W48 checkpoints, MIT-licensed."
lead: "HRNet is a convolutional network that keeps a high-resolution feature stream through repeated multi-scale fusion, instead of recovering resolution after downsampling. LibreYOLO wraps the official top-down pose variant for inference and validation."
keywords: [HRNet, human pose estimation, top-down pose, COCO-17 keypoints, high-resolution network]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # No person source given: HRNet pairs itself with a lightweight
        # LibreYOLO9t detector automatically and logs that choice once.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreHRNetw32-pose.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Person source
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Skip detection entirely: treat the whole image as one person.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Or hand HRNet boxes from a detector you already ran.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Or pair it with a specific LibreYOLO detector instead of the
        # LibreYOLO9t default.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # The exported graph is the fixed-canvas heatmap head alone: it takes
        # a batch of already-cropped, already-normalized person crops and
        # returns raw heatmaps. Person detection, crop geometry, heatmap
        # decoding and OKS suppression are not part of this graph; running it
        # outside LibreYOLO means reimplementing that decode step yourself.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
---

## Install

HRNet needs no extra beyond the base package.

```bash
pip install libreyolo
```

Its default person detector, a lightweight LibreYOLO9t checkpoint, downloads
automatically the first time HRNet pairs with it.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

HRNet is a top-down pose estimator: it needs a person box before the pose head
can run, so every call resolves one. Left alone, it pairs itself with a
LibreYOLO9t detector the first time and logs that choice. `cropped=True` skips
detection and treats the whole image as one person; `person_boxes` accepts
boxes from a detector you already ran; `person_detector` accepts `"auto"`,
`"rfdetr"`, any LibreYOLO detection model, or a plain callable. `flip_test=True`
runs the model on the horizontally flipped crop as well and averages the two
heatmaps, HRNet's own test-time augmentation; the generic `augment=True` is not
defined here. Multi-image sources run sequentially: HRNet's detector and
variable per-image person count do not support stacked prediction. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two sizes, `w32` and `w48`, both predicting the standard COCO-17 keypoint set
from a fixed-resolution person crop; `w48` is the wider of the two backbones.

The upstream model zoo reports pose accuracy for each size with its own person
detector, its own flip-testing setup, and the official COCO evaluation
protocol. LibreYOLO's default pairing uses a different detector, so a
validation run here measures that combination, not the upstream one; matching
the upstream figures needs the same person boxes, detector scores, and flip
setting the original evaluation used.

## Validate

`val()` runs COCO-style keypoint OKS-AP through `pycocotools` and accepts a
YOLO-pose `data.yaml` or a COCO keypoints JSON plus an images directory.

<code-tabs name="val" />

Validation drives HRNet's own `predict()` internally, so it uses whatever
person detector the model was built or called with. Construct the model with
an explicit `person_detector=` to keep that source fixed across runs, rather
than letting each call re-resolve the default.

## Export

<export-matrix />

HRNet's export contract covers ONNX, TorchScript, OpenVINO and TensorRT only;
any other format raises before the trace starts. Every export is the
fixed-canvas heatmap head alone, batch-one FP32, taking a person crop and
returning raw heatmaps: the affine crop geometry ahead of it and the heatmap
decoding, flip restoration and OKS suppression behind it stay in Python, so a
full image-in, keypoints-out pipeline still needs LibreYOLO on the other end.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
