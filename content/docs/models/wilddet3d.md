---
title: WildDet3D
families:
  - wilddet3d
seo_title: WildDet3D in LibreYOLO
description: WildDet3D predicts three-dimensional boxes from a single image.
lead: WildDet3D predicts three-dimensional boxes from a single image.
keywords:
  - WildDet3D
  - LibreYOLO
  - detect3d
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreWildDet3D, SAMPLE_IMAGE

        from PIL import Image

        import numpy as np


        # Requires the separately installed upstream runtime.

        model = LibreWildDet3D(device="cpu")

        # Approximate pinhole intrinsics for the original image size.

        # Replace with your camera's measured 3x3 calibration for real geometry.

        width, height = Image.open(SAMPLE_IMAGE).size

        f = max(width, height)

        intrinsics = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
        dtype=np.float32)

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

WildDet3D accepts text, box or point prompts and requires original-image camera intrinsics. Install its separate upstream runtime and provide `runtime_path` and `runtime_python` when they are outside the active environment.

`result.boxes3d` holds centers, dimensions and wxyz quaternions in camera coordinates, plus confidence, class and intrinsics. Plotting projects cuboids into the image. Training, validation, tracking and export are not supported. See [3D detection](/docs/tasks/3d-object-detection).

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
