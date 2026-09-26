---
title: FiftyOne integration
seo_title: FiftyOne integration | LibreYOLO
description: >-
  The FiftyOne integration transfers predictions and datasets between LibreYOLO
  and FiftyOne.
lead: >-
  The FiftyOne integration transfers predictions and datasets between LibreYOLO
  and FiftyOne.
keywords:
  - FiftyOne integration
  - LibreYOLO
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        import fiftyone as fo

        from libreyolo import SAMPLE_IMAGE

        from libreyolo.integrations.fiftyone import apply_model


        dataset = fo.Dataset()

        dataset.add_sample(fo.Sample(filepath=str(SAMPLE_IMAGE)))

        apply_model(dataset, "LibreYOLO9s.pt", label_field="predictions",
        device="cpu")
---

## Install

```bash
pip install "libreyolo[fiftyone]"
```

FiftyOne remains outside `all`. Its headless OpenCV package shares the `cv2` module with the core install. Use a separate environment or reinstall the GUI OpenCV package if display windows are needed. FiftyOne starts a local database and needs write access to its user directory.

## Apply a model

<code-tabs name="predict" />

`apply_model(dataset, model, label_field="predictions", batch_size=...)` accepts a loaded model or checkpoint name. It forwards prediction thresholds and image-size arguments. `to_fiftyone_model()` returns a wrapper for other FiftyOne model APIs.

## Label mapping

Detection writes boxes; instance segmentation attaches masks or polylines. OBB writes closed polylines. Pose writes keypoints and detections in separate fields. Classification writes the top-1 label. Coordinates normalize against the original image; tracking IDs become `Detection.index`.

## Dataset conversion

`to_fiftyone(data, split="val")` imports YOLO-layout or COCO-JSON dataset YAMLs. `from_fiftyone(view, output_dir, split="train", classes=...)` writes a dataset YAML and labels. Pass `classes=` to keep class IDs stable across filtered views.
