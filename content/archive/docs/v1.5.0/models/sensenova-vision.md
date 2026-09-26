---
title: SenseNova-Vision
families: [sensenovavision]
seo_title: "SenseNova-Vision in LibreYOLO: 7 tasks, one checkpoint"
description: "Use SenseNova-Vision in LibreYOLO for detection, segmentation, panoptic, pose, points, depth and OCR from one prompted generative checkpoint."
lead: "SenseNova-Vision is a unified multimodal model that casts vision tasks as prompted generation on a shared decoder: boxes, points, keypoints and OCR words come out as tagged text, and depth, mask and panoptic maps come out as images a decoder renders. LibreYOLO loads it through LibreVLM and supports seven tasks from the one 7B checkpoint."
keywords: [SenseNova-Vision, SenseTime, unified multimodal model, Bagel, prompted detection, dense perception, referring segmentation, panoptic segmentation]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() switches tasks on the same loaded model.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Referring segmentation and panoptic
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="segment")
        # Segmentation is referring: it needs a target phrase, not a class list.
        model.set_classes(["the person furthest to the right"])
        result = model.predict("street.jpg")
        mask = result.masks.data[0]

        model.set_task("panoptic")
        # With no custom vocabulary, panoptic falls back to the COCO panoptic
        # categories the checkpoint was tuned on.
        result = model.predict("street.jpg")
        segment_map = result.panoptic.data
        for segment in result.panoptic.segments_info:
            print(segment)
    - label: Points, pose and OCR
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # With no vocabulary set, pose falls back to "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
---

## Install

SenseNova-Vision needs its own extra, which pulls in `accelerate` for the big-model dispatch this checkpoint needs and, on non-macOS platforms, `bitsandbytes` for 4-bit loading.

```bash
pip install "libreyolo[sensenova]"
```

The checkpoint is mirrored on Hugging Face under LibreYOLO's own org and downloads automatically on first use; it is CC BY-NC 4.0, non-commercial use only, and the loader prints that notice before every automatic download. See Licensing below.

## Predict

<code-tabs name="predict" />

Every prediction is a diffusion decode over the shared Bagel-MoT backbone, so it is a capability model rather than a real-time one: expect noticeably higher per-image latency than a purpose-built detector or segmenter. `dtype="auto"` (the default) loads bf16 on a GPU with enough memory and falls back to 4-bit NF4 quantization elsewhere, which needs `bitsandbytes`; pass `dtype="bf16"` to force full precision on a large enough GPU. `noise_seed=42` at construction seeds the diffusion sampler for reproducible dense outputs; pass `noise_seed=None` to disable seeding.

The seven tasks share one loaded checkpoint: `set_task()` switches between them without reloading. `set_classes()` sets the active vocabulary; detection, points, pose and panoptic accept a class list, while segmentation is referring and needs exactly the phrase to isolate. Each task returns the standard `Results` object with a different payload populated: `boxes` for detect, `points` for point, `boxes` and `keypoints` for pose, `ocr` for OCR, `depth_map` for depth, `masks` for segment, and `panoptic` (with `segments_info`) for panoptic. See [prediction](/docs/predict) for sources, streaming and result handling.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
