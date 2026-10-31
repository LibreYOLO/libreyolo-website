---
title: GTR
families: [gtr]
seo_title: "GTR: Gated Token Recurrence in LibreYOLO"
description: "Use GTR in LibreYOLO for detection, instance and semantic segmentation, pose, oriented boxes and depth. Install, predict, train, validate and export."
lead: "A dense prediction model family whose backbone replaces softmax attention with gated linear attention, run as a recurrence over image tokens. LibreYOLO loads six of its tasks as one family, with the task carried by the checkpoint filename."
keywords: [GTR, Gated Token Recurrence, gated linear attention, object detection, instance segmentation, pose estimation, oriented bounding boxes, depth estimation, semantic segmentation]
last_verified: "1.6.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreGTRs.pt", device="cpu")
        result = model(SAMPLE_IMAGE)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreGTRs-seg.pt", device="cpu")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreGTRs-pose.pt", device="cpu")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)   # (people, 17, 2)
        print(result.boxes.conf)
    - label: Oriented boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Oriented boxes run at a fixed 1024 x 1024 input.
        model = LibreYOLO("LibreGTRs-obb.pt", device="cpu")
        result = model(SAMPLE_IMAGE)

        # One row per box: (cx, cy, w, h, theta) in pixels, theta in [0, pi).
        print(result.obb.xywhr)
    - label: Depth
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreGTRs-depth.pt", device="cpu")
        result = model(SAMPLE_IMAGE)

        # Relative inverse depth on the original canvas: higher values are closer.
        depth = result.depth_map.data
        print(depth.shape, float(depth.min()), float(depth.max()))
    - label: Semantic segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreGTRs-sem.pt", device="cpu")
        result = model(SAMPLE_IMAGE)

        # (H, W) class IDs on the original canvas; model.names maps them to labels.
        print(result.semantic_mask.data.unique())
        print(model.names)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs.pt", device="cpu")
        # Detection dataset YAML with YOLO box labels.
        model.train(data="path/to/your/dataset.yaml", epochs=30, device="cpu", workers=0)
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs-seg.pt", device="cpu")
        # Dataset YAML with YOLO polygon labels.
        model.train(data="path/to/your/dataset.yaml", epochs=30, device="cpu", workers=0)
    - label: Segmentation from detection weights
      language: python
      code: |
        from libreyolo import LibreGTR

        # Loads every layer from the detection checkpoint except the mask head,
        # which starts untrained. The factory LibreYOLO() does not take this flag.
        model = LibreGTR(
            "LibreGTRs.pt",
            size="s",
            task="segment",
            allow_detect_to_segment_transfer=True,
            device="cpu",
        )
        model.train(data="path/to/your/dataset.yaml", epochs=30, device="cpu", workers=0)
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Needs a single-class dataset whose YAML declares kpt_shape: [17, 3],
        # and imgsz left at the native 640.
        model = LibreYOLO("LibreGTRs-pose.pt", device="cpu")
        model.train(data="path/to/your/pose-dataset.yaml", device="cpu", workers=0)
    - label: Oriented boxes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Dataset YAML with YOLO oriented-box labels.
        model = LibreYOLO("LibreGTRs-obb.pt", device="cpu")
        model.train(data="path/to/your/obb-dataset.yaml", device="cpu", workers=0)
    - label: Depth
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Ground-truth depth in meters keeps the model's metric scale.
        model = LibreYOLO("LibreGTRs-depth.pt", device="cpu")
        model.train(data="path/to/your/depth-dataset.yaml", device="cpu", workers=0)
    - label: Semantic segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Images plus single-channel class-ID masks; imgsz sets the square crop.
        model = LibreYOLO("LibreGTRs-sem.pt", device="cpu")
        model.train(data="path/to/your/semantic.yaml", device="cpu", workers=0)
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Needs the lora extra. Detection, instance segmentation, pose and
        # oriented boxes accept lora=True; depth and semantic segmentation raise.
        model = LibreYOLO("LibreGTRs.pt", device="cpu")
        model.train(data="path/to/your/dataset.yaml", epochs=30, lora=True, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs.pt", device="cpu")
        metrics = model.val(data="path/to/your/dataset.yaml", workers=0)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs-seg.pt", device="cpu")
        metrics = model.val(data="path/to/your/dataset.yaml", workers=0)

        print(metrics["metrics/mAP50-95(M)"])   # masks
        print(metrics["metrics/mAP50-95(B)"])   # boxes
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs-pose.pt", device="cpu")
        metrics = model.val(data="path/to/your/pose-dataset.yaml", workers=0)

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Oriented boxes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Label IDs must follow the DOTA v1.0 class order of the checkpoint.
        model = LibreYOLO("LibreGTRs-obb.pt", device="cpu")
        metrics = model.val(data="path/to/your/obb-dataset.yaml", workers=0)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: Depth
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs-depth.pt", device="cpu")
        metrics = model.val(data="path/to/your/depth-dataset.yaml", workers=0)

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/delta1"])
    - label: Semantic segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs-sem.pt", device="cpu")
        metrics = model.val(data="path/to/your/semantic.yaml", workers=0)

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreGTRs.pt", device="cpu")
        model.export(format="onnx")          # static FP32 graph, 640 x 640
        model.export(format="torchscript")
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreGTRs.pt", device="cpu")
        path = model.export(format="onnx")

        # The factory routes on the file suffix, so the exported file loads
        # like a checkpoint and returns the same Results object.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

GTR needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

Adapter fine-tuning with `lora=True` needs the `lora` extra.

```bash
pip install "libreyolo[lora]"
```

On CPU and Apple silicon the backbone runs a portable PyTorch recurrence. On
CUDA it uses the `flash-linear-attention` kernels when that package is
importable, and the same portable recurrence when it is not. The package is not
a LibreYOLO extra. On a compatible CUDA system, install the version the library
documents.

```bash
pip install flash-linear-attention==0.5.0
```

The authors' custom CUDA operator and TensorRT plugin are not part of
LibreYOLO, and the portable recurrence does not reproduce the latency the paper
reports.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The checkpoint filename selects the task: no suffix for detection, then `-seg`,
`-pose`, `-obb`, `-depth` and `-sem`. Detection, instance segmentation, pose,
depth and semantic segmentation come in four sizes, `s`, `m`, `l` and `x`.
Oriented boxes come in `s` and `x`, the only sizes upstream publishes.

Every task returns the `Results` object all families return. Detection fills
`result.boxes`, instance segmentation adds `result.masks`, and pose adds
`result.keypoints` for one class, person, with the 17 COCO keypoints. Each pose
box is the extent of its keypoints, and keypoint visibility is reported as 1.
Detection, instance segmentation, pose and oriented boxes decode a fixed set of
queries by top-K selection with no NMS step, so `conf` and `max_det` filter the
output and `iou` has no effect.

Detection, instance segmentation, pose and depth take a square input, 640 by
default. `imgsz` must be a single integer that is a multiple of 32 and at least
160; a `(height, width)` pair raises a `ValueError`.

Oriented boxes run at a fixed 1024 by 1024 input. The image is resized to fit
without distortion and padded at the bottom and right. `result.obb.xywhr` holds
`(cx, cy, w, h, theta)` in pixels, with `w >= h` and `theta` in `[0, pi)`. The
published checkpoints predict the 15 DOTA v1.0 classes in upstream order.

Depth fills `result.depth_map` with relative inverse depth on the original
image canvas, so higher values are closer. The network itself predicts metric
depth, and `1 / result.depth_map.data` recovers that estimate in meters; the
meter scale holds only for cameras and scenes like its training data.
Preprocessing stretches the image to a square `imgsz`.

Semantic segmentation fills `result.semantic_mask` with an `(H, W)` map of
class IDs on the original canvas, over the 19 Cityscapes classes that
`model.names` lists. The image is letterboxed into a 1024 by 2048 canvas, and
the network averages overlapping 1024-pixel windows at a 768-pixel stride, as
upstream evaluates.

See [prediction](/docs/predict) for sources, streaming and result handling.

## Train

All six tasks train through `train()`, which reads the task from the loaded
checkpoint and applies that task's upstream recipe. Every recipe uses AdamW and
trains in FP32 by default (`amp=False`).

<code-tabs name="train" />

Detection defaults to 30 epochs, batch 4 and `lr0=5e-4`, with a 2000-iteration
warmup, six flat epochs and cosine decay. Mosaic (`mosaic_prob=0.5`) and batch
MixUp (`mixup_prob=0.5`) run for the first `mosaic_epochs=6` epochs, and the
strong augmentations switch off for the last two. Runs shorter than 30 epochs
compress these phases to fit. An `optimizer` other than `"adamw"` or a
`scheduler` other than `"flat_cosine"` raises a `ValueError` rather than being
ignored. The backbone trains at a size-specific fraction of `lr0`.

The other tasks follow their own upstream recipes:

- Instance segmentation keeps the detection schedule without Mosaic or MixUp
  and adds mask losses. Upstream starts it from the COCO detector. LibreYOLO
  does the same only when asked, through `allow_detect_to_segment_transfer=True`
  on `LibreGTR`, and the mask head then starts untrained. On the CLI,
  `task=segment` with a detection checkpoint makes the same transfer.
- Pose runs 92 epochs for `s` and `m` and 74 for `l` and `x`, with a
  500-iteration warmup and then a constant learning rate. It needs `imgsz=640`
  and a single-class dataset whose YAML declares `kpt_shape` with 17 keypoints;
  anything else raises before the run starts.
- Oriented boxes train at the fixed 1024 input for 20 epochs on `s` and 30 on
  `x`, with random flips (`flip_prob=0.75`) and rotations (`degrees=180`) in
  place of Mosaic and MixUp. A dataset with a different class count rebuilds the
  heads.
- Depth runs 60 epochs at batch 8 and `lr0=2e-4` with a SILog loss.
  Ground-truth depth in meters keeps the metric scale.
- Semantic segmentation runs 30 epochs at batch 8 on 1024-pixel square crops,
  and `imgsz` sets the crop size. Its head uses BatchNorm, and batches much
  smaller than 8 give noisy batch statistics.

`lora=True` freezes the backbone and the decoder layers and trains adapters on
the backbone's q, k and v projections and on the decoder layers' linear layers,
while the encoder and heads keep training. It applies to detection, instance
segmentation, pose and oriented boxes; depth and semantic segmentation raise a
`ValueError`. Export merges the adapters into dense weights. See
[LoRA](/docs/train/lora).

`pretrained=False` reinitializes the whole network and trains from scratch. Any
value other than `True` or `False` raises a `ValueError`; to start from other
weights, load them as the model. `resume=True` keeps writing into the checkpoint's own run
directory with the device passed to the call, and `exist_ok=False` opens a new
numbered run instead.

What has been checked: inference parity against the pinned upstream graph on
CPU for every published size of every task, and CPU training runs that
complete, save and reload for every task. What has not: convergence of a full
fine-tune on a real dataset, multi-GPU training, and reproduction of the
accuracy the authors report.

[Dataset setup](/docs/train/datasets) describes the required training data.

## Validate

`val()` returns a dictionary of `metrics/` keys. Detection reports box mAP under
`metrics/mAP50-95` and `metrics/mAP50`, and oriented boxes report rotated-box
mAP under the same keys. Instance segmentation reports masks under `(M)` and
boxes under `(B)`, and pose reports keypoint OKS metrics under
`metrics/keypoints_*`. Depth reports
`metrics/abs_rel`, `metrics/rmse` and `metrics/delta1` to `metrics/delta3`, and
semantic segmentation reports `metrics/mIoU` and `metrics/pixel_accuracy`.

<code-tabs name="val" />

Validating a published oriented-box checkpoint expects label IDs in its DOTA
v1.0 class order, so a dataset labeled in a different order needs its IDs
remapped first. Depth validation fits a per-image scale and shift in inverse-depth space
rather than following the upstream evaluation protocol, so its numbers are not
comparable to the paper's. [Validation](/docs/train/validation) explains the
dataset requirements and returned metrics.

## Export

<export-matrix />

ONNX and TorchScript are the two formats; any other raises
`NotImplementedError`. Both produce a static FP32 graph, and `dynamic=True` or
`half=True` raises a `ValueError`. Detection, instance segmentation, pose and
depth export at a fixed square input, oriented boxes at 1024 by 1024, and
semantic segmentation at a 1024 by 2048 canvas with the sliding windows inside
the graph. Depth exports at batch 1. The recurrence exports as an ONNX `Loop`.
For semantic segmentation, ONNX export skips graph simplification by default
because it is slow on CPU and does not shrink the graph; pass `simplify=True` to
run it anyway.

An exported artifact loads back through `LibreYOLO()` on its file suffix and
returns the same `Results`.

<code-tabs name="export" />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
