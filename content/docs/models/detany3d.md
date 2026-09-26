---
title: DetAny3D
families:
  - detany3d
seo_title: DetAny3D in LibreYOLO
description: DetAny3D predicts three-dimensional boxes from a single image.
lead: DetAny3D predicts three-dimensional boxes from a single image.
keywords:
  - DetAny3D
  - LibreYOLO
  - detect3d
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreDetAny3D, SAMPLE_IMAGE

        # Requires the separately installed upstream runtime.
        model = LibreDetAny3D(device="cpu")
        result = model.predict(SAMPLE_IMAGE, text="person")
        print(result.boxes3d)
---

## Install

```bash
pip install "libreyolo[hf]"
```

## Predict

<code-tabs name="predict" />

DetAny3D estimates camera intrinsics internally and accepts text, box or point prompts. Install its separate upstream runtime and its grounding dependencies. Default confidence is 0.37 and the text threshold is 0.25.

`result.boxes3d` holds centers, dimensions and wxyz quaternions in camera coordinates, plus confidence, class and intrinsics. Plotting projects cuboids into the image. Training, validation, tracking and export are not supported. See [3D detection](/docs/tasks/3d-object-detection).

## Licensing

<provenance-box></provenance-box>
