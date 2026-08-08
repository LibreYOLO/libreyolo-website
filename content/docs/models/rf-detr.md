---
title: RF-DETR
families: [rfdetr]
seo_title: "RF-DETR: train, fine-tune and export under MIT"
description: "Run RF-DETR for detection, instance segmentation, pose and oriented boxes in LibreYOLO. Benchmarks, checkpoints, training and export, all MIT-licensed."
lead: "A transformer detector that skips NMS entirely and still runs in real time. In LibreYOLO it is one of the two flagship families, so it gets every feature first: four tasks, LoRA fine-tuning, CUDA graphs and eight export targets."
keywords: [RF-DETR, real-time detection transformer, DETR, object detection, instance segmentation, pose estimation, oriented bounding boxes]
last_verified: "1.5.0"
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: "LibreRFDETRs running detection on video. Same model, same Results object, whether you call it from Python or the CLI."
snippets:
  quickstart:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Weights download from Hugging Face on first use
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
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=512,
            batch=8,
            lr0=1e-4,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Adapter-only fine-tune: far less memory, much smaller artifacts
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
        model.export(format="onnx", imgsz=512)          # LibreRFDETRs.onnx
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
    a: Yes. The LibreYOLO implementation is MIT licensed and the RF-DETR weights published in the LibreYOLO Hugging Face org are Apache-2.0. Neither license requires you to open source your application, and neither is copyleft.
  - q: Does RF-DETR need NMS?
    a: No. It predicts a fixed set of queries and selects the top scoring ones, so there is no non-maximum suppression step. The conf and max_det arguments still apply because they filter and cap that selection.
  - q: Why does my imgsz get rejected?
    a: The input resolution has to divide evenly by the backbone patch size multiplied by the window count. LibreYOLO validates this before the run starts and names the nearest two valid sizes in the error, so pick whichever one it suggests.
  - q: Should I pick RF-DETR or YOLOv9?
    a: RF-DETR usually wins on crowded scenes and overlapping objects because set prediction avoids NMS merge errors, and it covers four tasks. YOLOv9 is lighter, trains faster on small datasets, and exports to more edge formats including TFLite and ncnn. Both are flagship families, so both get new features first.
  - q: Can I train RF-DETR on a single GPU?
    a: Yes. Start from a COCO checkpoint, keep the default learning rate, and use batch=-1 to let autobatch pick a batch size that fits your card. For very small datasets, LoRA fine-tuning with lora=True cuts memory further.
related:
  - href: /docs/models/yolov9
    label: YOLOv9
    note: The other flagship. CNN, lighter, more export targets.
  - href: /docs/models/d-fine
    label: D-FINE
    note: Closest DETR sibling, detection and instance segmentation.
  - href: /docs/tasks/object-detection
    label: Object detection
    note: Every detector in the library, side by side.
  - href: /docs/export/onnx
    label: ONNX export
    note: The path most RF-DETR deployments take.
---

## Overview

RF-DETR is a real-time detection transformer. Instead of predicting a dense grid
of boxes and then deduplicating them, it predicts a fixed set of object queries
and matches them to ground truth one to one during training. There is no anchor
grid and no non-maximum suppression at inference, which is why its output stays
stable in crowded scenes where NMS-based detectors merge or drop overlapping
objects.

It is one of the two flagship families in LibreYOLO, alongside YOLOv9. In
practice that means new capabilities are designed against it first and fully
validated on GPU before they reach the rest of the library.

<tier-note>
Flagship families are the safest place to start. Every feature in the library is
built and GPU-validated here before it rolls out, and both flagships must
support a feature before it ships at all.
</tier-note>

RF-DETR is the only family in LibreYOLO that covers all four box-shaped tasks
with published weights.

<task-support />

## Benchmarks

<benchmark-table task="detect" />

<va-embed />

The accuracy gap between sizes is mostly resolution, not depth: every RF-DETR
variant carries a similar parameter count and scales by running at a larger
input. That makes the size choice a latency decision more than a memory one.

## Quickstart

RF-DETR needs its optional extra, because it pulls in a recent `transformers`
for the backbone.

```bash
pip install "libreyolo[rfdetr]"
```

Then load a checkpoint and predict. Weights download automatically the first
time you name one.

<code-tabs name="quickstart" />

The returned `Results` object is identical to the one every other family
returns, so swapping RF-DETR for a different detector is a one line change.

## Checkpoints

<checkpoint-table />

## Train on your own data

Fine-tuning from a COCO checkpoint is the normal path and usually converges in
far fewer epochs than training from scratch. RF-DETR is trainable for detection,
segmentation, pose and oriented boxes.

<code-tabs name="train" />

Two arguments matter more here than on CNN detectors. Keep `lr0` low, because
transformer detectors are sensitive to a high starting learning rate. And leave
`imgsz` at the value the checkpoint was trained for unless you have a reason to
change it, since resolution is how this family trades accuracy for speed.

For a walk through the shared training machinery, including datasets,
augmentation, multi-GPU and experiment loggers, see the
[training guide](/docs/train).

## Reproduce our numbers

Every number in the benchmark table above comes from a run you can repeat. This
is the exact command behind the nano row, against full COCO `val2017`.

<code-tabs name="reproduce" />

## Export

<export-matrix />

ONNX is the most travelled path and is parity validated for all four tasks.
TensorRT is validated for detection and experimental for the other three.

<code-tabs name="export" />

## How RF-DETR works

The backbone is a pretrained DINOv2 vision transformer, which is the main reason
the family transfers well to small custom datasets: the representation is
already strong before you show it a single label of your own.

Backbone features feed a deformable attention decoder. Rather than attending to
every spatial position, each query samples a small number of learned offsets,
which is what brings a DETR-style architecture into real-time latency. The
decoder emits a fixed number of queries per image; each carries a class
distribution and a box.

Training uses bipartite matching. A Hungarian matcher assigns each prediction to
at most one ground truth object, so duplicate predictions are penalized during
training instead of being cleaned up afterward at inference. The segmentation,
pose and oriented box variants keep this decoder and attach a task head to it,
which is why one family covers four tasks without four separate architectures.

Resolution is constrained: the input has to divide evenly by the backbone patch
size multiplied by the window count. LibreYOLO checks this before a run starts
and names the nearest valid sizes rather than failing deep inside the forward
pass.

## Provenance and licensing

<provenance-box>
LibreYOLO's RF-DETR is a port of the upstream Apache-2.0 release, adapted to the
shared model, trainer and export contracts. Architecture and weight
compatibility are preserved, so upstream checkpoints convert cleanly.
</provenance-box>

<license-answer q="What license does RF-DETR use?">
Two licenses apply and they are independent. The code in LibreYOLO is MIT. The
weights published in the LibreYOLO Hugging Face org are Apache-2.0, inherited
from the upstream release. Both are permissive, and neither obliges you to
release your own source.
</license-answer>

<license-answer q="Can I use RF-DETR commercially?">
Yes, including in closed source and commercial products, under both the MIT code
license and the Apache-2.0 weights. Apache-2.0 asks you to keep the license and
attribution notices with any copy of the weights you redistribute. If you train
your own weights on your own data, the resulting checkpoint is yours.
</license-answer>

## Citation

If RF-DETR contributes to your research, cite the original work.

```text
@article{rfdetr2025,
  title   = {RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  author  = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  journal = {arXiv preprint arXiv:2511.09554},
  year    = {2025}
}
```
