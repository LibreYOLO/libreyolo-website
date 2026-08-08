---
title: Gaze estimation
seo_title: "Gaze estimation in LibreYOLO"
description: "Estimate per-face gaze pitch and yaw in LibreYOLO. Predict from Python or the CLI, read angles in radians, and export the gaze head to ONNX."
lead: "Gaze estimation returns a look direction for every face in an image. LibreYOLO models it as a two-stage task: a face detector runs first, and a gaze head reads pitch and yaw from each face crop it returns."
keywords: [gaze estimation python, eye tracking, pitch yaw gaze, L2CS-Net, gaze direction, head pose, libreyolo gaze task]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # With no face_detector given, prediction falls back to OpenCV's
        # bundled detector, so nothing downloads beyond the checkpoint.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # radians, one row per face
        print(gaze.pitch_deg, gaze.yaw_deg)      # the same angles in degrees
        print(gaze.direction_3d)                 # (N, 3) unit vectors
    - label: CLI
      language: bash
      code: |
        # Unlike the Python path, the CLI has no automatic fallback: gaze
        # models require an explicit face detector, and it must be a
        # LibreYOLO detector whose boxes are faces.
        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg face_detector=face-detector.pt save=True
    - label: Choose the face source
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Hand the gaze head boxes from a detector you already ran.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Or name one of the bundled detectors.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
---

## Definition

Gaze estimation returns two angles per face. `result.gaze` is a `Gaze` payload
of shape `(N, 2)`, column 0 pitch and column 1 yaw, in radians, aligned row by
row with `result.boxes`, the detected face boxes. The convention is the one
L2CS-Net uses: positive yaw rotates the gaze toward the subject's left,
positive pitch rotates it downward.

The same payload exposes `pitch_deg` and `yaw_deg` for degrees, and
`direction_3d`, an `(N, 3)` unit vector in the camera frame with columns
`(x, y, z)`.

Because the task is two-stage, a prediction depends on two models. Faces the
detector misses have no gaze row, and boxes it places badly produce angles from
a badly cropped face. The canonical task key is `gaze`; `gaze-estimation`
normalizes to it.

## Models

[L2CS-Net](/docs/models/l2cs) is the only family serving this task. It pairs a
ResNet trunk with two parallel angle-bin classification heads, one for pitch and
one for yaw, over 448x448 face crops. Five backbone depths are supported
architecturally, and one, the ResNet-50, has a published checkpoint.

The weights carry a real restriction. They are trained on Gaze360, whose license
permits research and non-commercial use only and forbids redistribution, so
LibreYOLO mirrors nothing for this family. The one checkpoint the library can
fetch automatically comes straight from the authors' own Google Drive
distribution, over `gdown`, after printing the license terms. Read
[L2CS-Net](/docs/models/l2cs) before deploying it.

```bash
pip install "libreyolo[gaze]"
```

Without that extra the library prints manual download instructions instead of
attempting the transfer. Predicting on and exporting a checkpoint you already
hold needs no extra at all.

## Predict

<code-tabs name="predict" />

The face source is chosen in one of three ways. `face_boxes` passes boxes you
already computed and skips detection. `face_detector` accepts `"auto"`,
`"haar"`, `"yunet"`, a LibreYOLO detection model, or a plain callable, and can
be set on the constructor or per call. Left unset in Python, prediction falls
back to OpenCV's bundled detector, Haar on OpenCV 4 and YuNet on OpenCV 5, so a
bare call works offline.

The CLI does not share that fallback. `libreyolo predict` rejects a gaze model
without `face_detector=`, and the value it takes is a LibreYOLO detector name or
checkpoint path. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Train

No family in this task trains inside LibreYOLO. `LibreL2CS.train()` raises:
train at the upstream L2CS-Net project and load the resulting state dict here.

## Validate

Validation against gaze ground-truth datasets is out of scope, and `val()`
raises rather than returning metrics it did not compute. There is no `metrics/`
dictionary for this task. Evaluate upstream, on the dataset the checkpoint was
trained for.

## Export

<code-tabs name="export" />

The gaze export contract covers ONNX, TorchScript, ExecuTorch, TensorRT and
OpenVINO. What leaves the library is the ResNet trunk and the two angle-bin
heads alone: the graph takes a preprocessed 448x448 face crop and returns raw
yaw and pitch logits. Face detection, cropping, the softmax, the bin
expectation and the conversion to angles all stay in Python, in
`libreyolo.models.l2cs.utils`. See [export](/docs/export) for the formats and
their arguments.
