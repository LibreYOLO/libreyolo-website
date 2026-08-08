---
title: Inference performance
seo_title: "Faster inference in LibreYOLO"
description: "CUDA graphs, half precision, batching, tiled inference and test-time augmentation at predict time, with the real defaults and which families support each."
lead: "Five prediction-time controls change throughput or accuracy: CUDA graph replay, precision, batching, tiling and test-time augmentation. Each applies to a specific set of families, and two of them cost accuracy or latency rather than saving it."
keywords:
  - cuda graphs pytorch inference
  - yolo batch inference python
  - fp16 inference
  - tiled inference small objects
  - sahi slicing
  - test time augmentation detection
  - capture_graph
  - batch predict directory
last_verified: "1.5.0"
verification: "Argument defaults from InferenceRunner.__call__ in libreyolo/models/base/inference.py. CUDA graph API from BaseModel.capture_graph, graph_info, release_graphs and cuda_graph_scope in libreyolo/models/base/model.py; family opt-in from the SUPPORTS_CUDA_GRAPH class variable. Half-precision behavior from NOOP_PREDICT_KWARGS in libreyolo/utils/predict_args.py, the CLI warning in libreyolo/cli/commands/predict.py, and CAST_RECIPES plus SUPPORTED_FAMILIES in libreyolo/quant/api.py. Batching conditions from InferenceRunner._process_in_batches and _predict_batch. Tiling from _predict_tiled and _merge_tile_detections. Test-time augmentation from BaseModel._predict_augment and _merge_tta, with TTA_ENABLED, TTA_SCALES and TTA_FIXED_SIZE read across libreyolo/models/."
snippets:
  batch:
    - label: Batched inference over a folder
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # One stacked forward per chunk of 4 on families that support it.
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: Streaming, so the list never materializes
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: Capture up front, then replay (needs CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Pay warmup and capture once, off the first request.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Capture only once a shape repeats (needs CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" waits for a shape to be seen twice, so one-shot work
        # never pays for capture.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Install the export extra
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Export and load back, at the default precision
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 export (build and run this on a CUDA machine)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 in PyTorch, via a cast recipe (needs CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # A cast recipe reads no calibration data.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Tiled inference on a large image
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tiling only engages when the image is larger than the input size.
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Test-time augmentation
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
---

## The controls and their defaults

Every one of these is an argument to `predict`, and every default is off.

| Argument | Default | Effect |
|---|---|---|
| `batch` | `1` | Images per forward pass, for folder and list sources |
| `cuda_graph` | `False` | Replay the forward from a captured CUDA graph |
| `tiling` | `False` | Split a large image into overlapping tiles |
| `overlap_ratio` | `0.2` | Tile overlap when `tiling` is on |
| `augment` | `False` | Run flipped views and merge them |
| `half` | | Accepted, warned, and ignored |
| `device` | `None` | Move the model before predicting |

`imgsz` also affects cost, since it sets the resolution the model runs at, but
it is an accuracy argument first and belongs with the model rather than here.

## Batching

<code-tabs name="batch" />

`batch` applies to folder and list sources. With `batch=1`, images run one
forward pass each. Above `1`, each chunk is preprocessed, stacked into a single
tensor, run once, then sliced back so every family's existing single-image
postprocess sees what it expects.

The stacked path is taken only when all of these hold:

- `batch` is greater than `1`
- `tiling` is off
- test-time augmentation is not active
- the family sets `SUPPORTS_BATCHED_PREDICT`
- the underlying network is not in training mode

The last condition is not a technicality. A network in training mode would
normalize the stacked chunk with cross-image batch statistics, letting images in
the same chunk change each other's predictions, so those runs stay sequential.

`SUPPORTS_BATCHED_PREDICT` defaults to true. These families opt out and run one
image per forward regardless of `batch`: Depth Anything V2, Depth Anything 3,
EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5,
Real-ESRGAN, RetinaNet, SAM 3D Body, SwinIR, YOLOv1, ZipDepth, every
open-vocabulary detector, and every vision language model.

There is one more fallback. If preprocessing does not return uniform
`(1, C, H, W)` tensors of matching shape, dtype and device across the chunk, the
chunk runs sequentially rather than stacking, so correctness never depends on
the images happening to be the same size.

Combine `batch` with `stream=True` on a large folder to get batched forwards
without holding every result in memory.

## CUDA graphs

<code-tabs name="graphs" />

A CUDA graph records a forward pass once and replays it as a single launch.
Small detectors spend a large share of batch-1 time launching kernels, so
collapsing those launches is a throughput win, and replay output is bit
identical to eager execution.

`cuda_graph` takes three values. `False` is the default and does nothing.
`True` captures on first use for each input shape. `"auto"` waits until a shape
repeats before capturing, so one-shot and shape-varying work never pays the
capture cost.

`capture_graph(imgsz=None, batch=1, dtype=None)` moves that cost off the first
request. A graph is valid only for the exact shape it captured, so `batch` here
has to match how `predict` is later called.

`graph_info()` reports the captured graphs, replay counts, and any reason the
run fell back to eager. `release_graphs()` frees them and their static buffers.

Capture requires CUDA and a family that has opted in through
`SUPPORTS_CUDA_GRAPH`, because it needs a forward with no host-visible work and
that is verified per family. Asking for it on a family that has not opted in
raises `NotImplementedError` rather than silently running eager.

A graph records memory addresses, not values, so anything that relocates
parameters drops it. Changing device through `predict(device=...)`, quantizing
and dequantizing all invalidate captured graphs.

## Precision

<code-tabs name="precision" />

`half=True` at predict time does nothing. It is accepted for command line
compatibility, raises a warning saying it is a no-op, and is discarded before
it reaches any family. The CLI's `--half` flag prints the same warning for a
`.pt` model.

There are two real routes to lower precision.

For an exported artifact, precision is chosen at export time with
`export(format=..., half=True)`, and the resulting file loads back through
`LibreYOLO()` unchanged.

For PyTorch execution, `model.quantize(recipe="fp16")` casts the model to
float16 and installs hooks that keep float32 at the model's inputs and outputs.
`"bf16"` does the same with bfloat16. Neither cast reads calibration data, so
`calib` is ignored for them. Quantization currently covers four families:
YOLOv9, RF-DETR, BiRefNet and FeyNobg. A cast on a CPU device logs a warning
that it will be slow, so these recipes are meant for a GPU.

Both routes change numerics. Neither is a drop-in guarantee of the same
detections, so validate before deploying.

## Tiled inference

<code-tabs name="tiling" />

Tiling crops a large image into overlapping square tiles, predicts on each, and
merges the results. It is the option for small objects in high-resolution
images, where a whole-image resize shrinks the targets below what the model can
resolve.

Tile size is the model's input size, or `imgsz` when given, and it has to be
square. `overlap_ratio` defaults to `0.2`. Tiles that overlap are reconciled
with per-class non-maximum suppression at the `iou` threshold, and the merged
list is then truncated to `max_det`. This means `iou` has an effect on tiled
predictions even for families that run no NMS of their own.

Tiling is skipped, not merely cheap, when the image already fits: if both
dimensions are at or below the input size, one ordinary forward runs instead.
It is also skipped for classification, semantic segmentation and the `embed`
task, which fall back to a single pass because tiling has no meaning there.

It raises for tasks whose payload cannot be stitched back together: instance
segmentation masks, oriented boxes, points, depth, edges and normals. It cannot
be combined with `augment`.

The result carries `result.tiled` and `result.num_tiles`. With `save=True`,
tiled runs write a directory under `runs/tiled_detections` holding every tile,
the annotated image, a grid visualization, and a `metadata.json` recording the
tile size, overlap and thresholds, with `result.tiles_path` and
`result.grid_path` pointing at them.

## Test-time augmentation

<code-tabs name="tta" />

`augment=True` runs the image more than once and merges the detections with
per-class non-maximum suppression at the `iou` threshold. Like tiling, this
makes `iou` load-bearing for families that otherwise ignore it.

In practice this is horizontal flipping. The scale list `TTA_SCALES` defaults to
a single scale of `1.0` and no shipped family overrides it, so every family runs
two passes: the original image and its mirror. Families marked `TTA_FIXED_SIZE`
resize to a fixed square, which makes multi-scale a no-op for them in any case.

Semantic and panoptic segmentation take a different merge. Their flipped view is
flipped back and the two softmax distributions are averaged before the argmax,
rather than being merged as boxes.

Test-time augmentation is not available for every task. It raises for oriented
boxes, pose, points, depth, normals, edges, restoration, OCR and embedding
models, and cannot be combined with tiling.

These families disable it outright, so `augment=True` runs a single ordinary
pass: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS,
NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED,
every SAM variant, every open-vocabulary detector, and every vision language
model.

## Measuring

Nothing on this page carries a latency number, because a millisecond without
its hardware, runtime, precision and batch size is not a fact. Measured figures
across hardware and runtimes are published at
[visionanalysis.org](https://www.visionanalysis.org), and `libreyolo profile`
measures a specific model on the machine in front of you.
