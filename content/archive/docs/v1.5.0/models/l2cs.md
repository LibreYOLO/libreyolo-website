---
title: L2CS-Net
families: [l2cs]
seo_title: "L2CS-Net: gaze estimation in LibreYOLO"
description: "Use L2CS-Net in LibreYOLO for two-stage gaze pitch/yaw estimation. Install, predict and export; the Gaze360 checkpoint is research-only."
lead: "L2CS-Net is a two-stage gaze estimator: a face detector locates faces, and a ResNet trunk with two angle-bin classification heads predicts pitch and yaw per face. LibreYOLO wraps it for inference only."
keywords: [L2CS-Net, gaze estimation, eye tracking, pitch yaw, Gaze360, face detection]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # No face_detector given: falls back to OpenCV's bundled face
        # detector (Haar on OpenCV 4, YuNet on OpenCV 5), so this runs with
        # no extra download beyond the L2CS checkpoint itself.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreL2CSr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Face source
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Hand L2CS boxes from a detector you already ran.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Or name a specific bundled face detector.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # The exported graph is the ResNet trunk and the two angle-bin heads
        # alone: it takes a preprocessed 448x448 face crop and returns raw
        # (yaw_logits, pitch_logits), not decoded angles. The softmax,
        # bin-expectation and degree conversion stay in Python; see
        # libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
---

## Install

L2CS-Net needs no extra to construct, predict on, or export a model you
already have a checkpoint for.

```bash
pip install libreyolo
```

The one checkpoint LibreYOLO can fetch automatically, a Gaze360-trained
ResNet-50, downloads over `gdown` rather than a plain HTTP mirror, because it
lives on the author's Google Drive rather than the LibreYOLO org. That path
needs the `gaze` extra:

```bash
pip install "libreyolo[gaze]"
```

Without it, LibreYOLO prints manual download instructions instead of failing
silently.

## Predict

<code-tabs name="predict" />

L2CS-Net is a two-stage estimator: a face detector runs first, and the gaze
head reads pitch and yaw from each face crop it returns. Left alone, prediction
falls back to OpenCV's bundled detector, so a bare call works with no
additional download once the L2CS checkpoint itself is in hand. `face_boxes`
accepts boxes from a detector you already ran; `face_detector` accepts
`"auto"`, `"haar"`, `"yunet"`, a LibreYOLO detection model, or a plain
callable. `result.gaze` carries pitch and yaw in radians, aligned row by
row with `result.boxes`, the detected face boxes. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Five backbone depths share one input resolution and take the same arguments.
Gaze360, the dataset behind the only published checkpoint, trained a
ResNet-50; the other four depths are supported architecturally but have no
published weights to load.

## Export

<export-matrix />

<code-tabs name="export" />

## Licensing

<provenance-box>

LibreYOLO does not host or mirror any L2CS checkpoint: nothing for this family
exists in the LibreYOLO Hugging Face org, unlike most other families on this
site. The one checkpoint the library can fetch automatically comes straight
from the author's own Google Drive distribution, gated behind the Gaze360
license notice printed before the transfer starts, and is not the "republished
at huggingface.co/LibreYOLO" copy the summary above implies.

</provenance-box>
