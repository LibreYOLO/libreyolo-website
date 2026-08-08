---
title: RF-DETR
families: [rfdetr]
seo_title: "RF-DETR: train, fine-tune and export under MIT"
description: "Use RF-DETR in LibreYOLO for detection, instance segmentation, pose and oriented boxes. Install, predict, train, validate and export, all MIT-licensed."
lead: "A detection transformer that predicts a fixed set of objects instead of a dense grid, so it needs no NMS at inference. LibreYOLO supports it for four tasks."
keywords: [RF-DETR, real-time detection transformer, DETR, object detection, instance segmentation, pose estimation, oriented bounding boxes]
last_verified: "1.5.0"
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: "LibreRFDETRs, detection on video at 512 px."
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs.pt source=bus.jpg save=True
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics.box.map)      # mAP 50-95
        print(metrics.box.map50)    # mAP 50
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Reproduce COCO
      language: bash
      code: |
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384
      expect: |
        Class     Images  Instances     mAP50   mAP50-95
        all         5000     36335      0.699      0.514
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512
    - label: Run the ONNX
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession("LibreRFDETRs.onnx")
        dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
        boxes, logits = session.run(None, {session.get_inputs()[0].name: dummy})
---

## Install

RF-DETR needs its own extra, which pulls in `transformers` for the backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the query
selection; there is no NMS step to tune. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Four sizes, and four tasks that share one architecture: segmentation, pose and
oriented boxes reuse the detection decoder with a different head, so they take
the same arguments. Sizes differ mainly by input resolution rather than depth,
so choosing one is a latency decision more than a memory one.

<benchmark-table task="detect" />

<va-embed />

## Train

Fine-tuning from a COCO checkpoint converges in far fewer epochs than training
from scratch, and works for all four tasks.

<code-tabs name="train" />

Two arguments matter more here than on a CNN detector. Keep `lr0` at or below
`1e-4`, since transformer detectors diverge at learning rates a YOLO model
tolerates. Leave `imgsz` at the checkpoint's native resolution unless you have a
reason to change it. The input must divide evenly by the backbone patch size
times the window count; LibreYOLO checks this before the run starts and names
the nearest valid sizes.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` reports mAP 50-95, mAP 50 and per-class results against any dataset in
the same format you trained on.

<code-tabs name="val" />

The last command reproduces the first row of the benchmark table above against
full COCO `val2017`.

## Export

<export-matrix />

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />

