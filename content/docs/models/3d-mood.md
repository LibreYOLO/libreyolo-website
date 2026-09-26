---
title: 3D-MOOD
families:
  - 3dmood
seo_title: 3D-MOOD in LibreYOLO
description: 3D-MOOD predicts three-dimensional boxes from a single image.
lead: 3D-MOOD predicts three-dimensional boxes from a single image.
keywords:
  - 3D-MOOD
  - LibreYOLO
  - detect3d
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import Libre3DMOOD, SAMPLE_IMAGE

        import numpy as np


        # Requires the separately installed upstream runtime.

        model = Libre3DMOOD(device="cpu")

        # Use measured calibration for the original image, before resizing.

        intrinsics = np.load(input("Camera intrinsics .npy path: "),
        allow_pickle=False)

        result = model.predict(SAMPLE_IMAGE, intrinsics=intrinsics,
        text="person")

        print(result.boxes3d)
---

## Install

```bash
pip install "libreyolo[hf]"
```

## Predict

<code-tabs name="predict" />

3D-MOOD performs open-set detection using text labels and original-image camera intrinsics. Install its separate upstream runtime and provide `runtime_path` and `runtime_python` when needed.

`result.boxes3d` holds centers, dimensions and wxyz quaternions in camera coordinates, plus confidence, class and intrinsics. Plotting projects cuboids into the image. Training, validation, tracking and export are not supported. See [3D detection](/docs/tasks/3d-object-detection).

## Licensing

<provenance-box></provenance-box>
