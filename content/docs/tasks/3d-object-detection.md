---
title: 3D object detection
seo_title: 3D object detection | LibreYOLO
description: >-
  3D detection returns object centers, dimensions and orientations in camera
  coordinates.
lead: >-
  3D detection returns object centers, dimensions and orientations in camera
  coordinates.
keywords:
  - 3D object detection
  - LibreYOLO
last_verified: 1.6.0
---

## Models

[WildDet3D](/docs/models/wilddet3d), [3D-MOOD](/docs/models/3d-mood), [FCOS3D](/docs/models/fcos3d) and [DetAny3D](/docs/models/detany3d) return `Results.boxes3d`. FCOS3D runs natively; the other adapters require their separate upstream runtimes.

## Predict

Use the model page's calibrated example. WildDet3D, 3D-MOOD and FCOS3D require a measured 3x3 intrinsic matrix for the original image. DetAny3D estimates intrinsics internally. Box orientations use wxyz quaternions. `result.plot()` projects cuboids into the source image.

## Train

These four adapters do not support training, validation, tracking or export.
