---
title: RF-DETR
families: [rfdetr]
seo_title: "RF-DETR: train, fine-tune and export under MIT"
description: "Run RF-DETR for detection, instance segmentation, pose and oriented boxes in LibreYOLO. Benchmarks, checkpoints, training and export, all MIT-licensed."
lead: "A detection transformer that predicts a fixed set of objects instead of a dense grid, so it needs no NMS at inference. LibreYOLO supports it for four tasks."
keywords: [RF-DETR, real-time detection transformer, DETR, object detection, instance segmentation, pose estimation, oriented bounding boxes]
last_verified: "1.5.0"
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: "LibreRFDETRs, detection on video at 512 px."
snippets:
  quickstart:
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
  reproduce:
    - label: CLI
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
faq:
  - q: Is RF-DETR free for commercial use?
    a: Yes. The implementation is MIT and the published weights are Apache-2.0. Neither is copyleft, so nothing obliges you to release your own source.
  - q: Does RF-DETR need NMS?
    a: No. It predicts a fixed set of queries and keeps the highest scoring ones. conf and max_det still apply, because they filter that selection.
  - q: Why was my imgsz rejected?
    a: The input must divide evenly by the backbone patch size times the window count. LibreYOLO checks before the run starts and names the nearest two valid sizes.
  - q: RF-DETR or YOLOv9?
    a: RF-DETR handles crowded and overlapping objects better, because there is no NMS to merge them, and it covers four tasks. YOLOv9 is lighter, trains faster on small datasets, and reaches more edge formats including TFLite and ncnn.
  - q: Can I train it on one GPU?
    a: Yes. Fine-tune from a COCO checkpoint, keep lr0 low, and set batch=-1 to let autobatch size it. LoRA reduces memory further.
related:
  - href: /docs/models/yolov9
    label: YOLOv9
    note: The other flagship. CNN, lighter, more export targets.
  - href: /docs/models/d-fine
    label: D-FINE
    note: Nearest DETR sibling. Detection and instance segmentation.
  - href: /docs/tasks/object-detection
    label: Object detection
    note: Every detector in the library, compared.
  - href: /docs/export/onnx
    label: ONNX export
    note: The path most deployments take.
---

## Tasks

RF-DETR is the only family that covers all four box-shaped tasks with published
weights. Segmentation, pose and oriented boxes reuse the detection decoder and
attach a task head, so they behave the same way at the API.

<task-support />

## Benchmarks

<benchmark-table task="detect" />

Accuracy scales with input resolution rather than depth here: all four
checkpoints sit within 3.5 M parameters of each other, so choosing a size is a
latency decision, not a memory one.

<va-embed />

## Quickstart

RF-DETR needs its own extra, which pulls in `transformers` for the backbone.

```bash
pip install "libreyolo[rfdetr]"
```

<code-tabs name="quickstart" />

Weights download on first use. The returned `Results` object is the same one
every other family returns, so swapping detectors is a one line change.

## Checkpoints

<checkpoint-table />

## Training

Fine-tuning from a COCO checkpoint converges in far fewer epochs than training
from scratch, and works for all four tasks.

<code-tabs name="train" />

Two arguments matter more here than on a CNN detector. Keep `lr0` at or below
`1e-4`, since transformer detectors diverge at learning rates a YOLO model
tolerates. Leave `imgsz` at the checkpoint's native resolution unless you have a
reason to change it, because resolution is how this family trades accuracy for
latency. See [training](/docs/train) for datasets, augmentation, multi-GPU and
loggers.

## Reproducing the numbers

The benchmark table above is reproducible. This is the exact command behind the
first row, against full COCO `val2017`.

<code-tabs name="reproduce" />

## Export

<export-matrix />

<code-tabs name="export" />

## Architecture

The backbone is a pretrained DINOv2 vision transformer, which is the main reason
the family transfers well to small custom datasets: the representation is
already strong before it sees a label of yours.

Backbone features feed a deformable attention decoder. Each query samples a
small number of learned offsets rather than attending to every spatial position,
which is what brings a DETR into real-time latency. The decoder emits a fixed
number of queries, each carrying a class distribution and a box.

Training uses bipartite matching: a Hungarian matcher assigns each prediction to
at most one ground truth object, so duplicates are penalized during training
rather than removed afterwards. That is why there is no NMS at inference, and
why crowded scenes stay stable.

Resolution is constrained. The input must divide evenly by the backbone patch
size times the window count, and LibreYOLO validates this before a run starts
rather than failing inside the forward pass.

## Licensing

<provenance-box>
The LibreYOLO implementation is a port of the upstream Apache-2.0 release,
adapted to the shared model, trainer and export contracts. Architecture and
weight compatibility are preserved, so upstream checkpoints convert cleanly.
</provenance-box>

### Can I use RF-DETR commercially?

Yes, in closed source and commercial products, under both licenses. Apache-2.0
asks you to keep its license and attribution notices with any copy of the
weights you redistribute; it does not reach your application code. Weights you
train yourself on your own data are yours.

## Citation

```text
@article{rfdetr2025,
  title   = {RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  author  = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  journal = {arXiv preprint arXiv:2511.09554},
  year    = {2025}
}
```
