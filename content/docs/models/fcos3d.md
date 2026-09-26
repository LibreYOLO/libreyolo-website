---
title: FCOS3D
families:
  - fcos3d
seo_title: FCOS3D in LibreYOLO
description: FCOS3D predicts three-dimensional boxes from a single image.
lead: FCOS3D predicts three-dimensional boxes from a single image.
keywords:
  - FCOS3D
  - LibreYOLO
  - detect3d
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreFCOS3D, SAMPLE_IMAGE

        import numpy as np


        model = LibreFCOS3D(device="cpu")

        # Use measured calibration for the original image, before resizing.

        intrinsics = np.load(input("Camera intrinsics .npy path: "),
        allow_pickle=False)

        result = model.predict(SAMPLE_IMAGE, intrinsics=intrinsics)

        print(result.boxes3d)
---

## Install

```bash
pip install "libreyolo[hf]"
```

## Predict

<code-tabs name="predict" />

FCOS3D runs natively on CPU or CUDA and requires original-image camera intrinsics. MPS is not supported. The default confidence is 0.05, rotated NMS IoU is 0.8 and `max_det` is 200.

`result.boxes3d` holds centers, dimensions and wxyz quaternions in camera coordinates, plus confidence, class and intrinsics. Plotting projects cuboids into the image. Training, validation, tracking and export are not supported. See [3D detection](/docs/tasks/3d-object-detection).

## Licensing

<provenance-box></provenance-box>
