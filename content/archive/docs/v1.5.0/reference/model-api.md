---
title: Model API
seo_title: "LibreYOLO model object methods and signatures"
description: "Every method on a loaded LibreYOLO model: predict, embed, track, val, train, export, save, quantize, info and the CUDA graph controls, with real defaults."
lead: "A loaded LibreYOLO model is an instance of BaseModel. This page lists the methods that instance carries, with the signatures and defaults read from libreyolo/models/base/model.py."
keywords:
  - libreyolo model methods
  - libreyolo predict arguments
  - libreyolo val arguments
  - libreyolo export arguments
  - model.track
  - model.quantize
  - capture_graph
last_verified: "1.5.0"
verification: "Signatures and defaults read from libreyolo/models/base/model.py and libreyolo/models/base/inference.py at v1.5.0. Family classes may narrow or extend these; train() is defined per family and only its shared cfg= wrapper is documented here."
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True returns a generator, one Results per frame or image.
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
---

## Construction

The factory returns a family class instance. Constructing that class directly
takes the same arguments, except that `size` is required:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` selects CUDA when available, then MPS, then CPU. An integer or
a digit string is read as a CUDA ordinal, so `device=0` and `device="0"` both
mean `cuda:0`. `task` is validated against the family's `SUPPORTED_TASKS`.
Passing `model_path=None` builds the architecture and leaves it in training
mode; passing a `dict` loads that state dict directly.

## predict and \_\_call\_\_

`predict` is an alias for `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | `None` | Image, list or tuple of in-memory images, directory, video file, or a screen source such as `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Confidence threshold |
| `iou` | `0.45` | IoU threshold for NMS |
| `imgsz` | `None` | Input size override; `None` uses the model's native size |
| `device` | `None` | Device override for this call |
| `classes` | `None` | Keep only these class IDs |
| `max_det` | `300` | Maximum detections per image |
| `augment` | `False` | Test-time augmentation |
| `save` | `False` | Write an annotated image or video |
| `batch` | `1` | Images per forward pass for directory and list sources |
| `stream` | `False` | Return a generator instead of a materialized list |
| `stream_buffer` | `False` | Keep every captured live frame instead of only the newest |
| `vid_stride` | `1` | Process every N-th video or screen frame |
| `show` | `False` | Display annotated frames in a window |
| `output_path` | `None` | Output path when `save=True` |
| `color_format` | `"auto"` | Color format hint for in-memory arrays |
| `tiling` | `False` | Tiled inference for large images |
| `overlap_ratio` | `0.2` | Tile overlap ratio |
| `output_file_format` | `None` | `"jpg"`, `"png"` or `"webp"` |
| `cuda_graph` | `False` | `True` captures on first use per input shape, `"auto"` waits for a shape to repeat |

A single image source returns one `Results`. A list, a tuple or a directory
returns a list of them, and `stream=True` returns a generator in every case.

Live stream sources are unbounded and require `stream=True`. `tiling` and
`augment` cannot be combined. Test-time augmentation raises for the `embed`,
`point` and `edge` tasks.

<code-tabs name="usage" />

With `batch > 1`, families whose `SUPPORTS_BATCHED_PREDICT` is true run one
stacked forward per chunk; `batch=1` keeps one forward per image.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

A convenience wrapper over `predict` that stacks every embedding row into a
single `(N_total, D)` tensor. The model must have been constructed with
`task="embed"`, otherwise it raises `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Yields one `Results` per frame with `track_id` set. `tracker` is
`"bytetrack"`, `"botsort"`, `"ocsort"` or `"deepocsort"`, and is ignored when
`tracker_config` is given because the config type selects the tracker.
`track_conf` maps to `track_high_thresh` for ByteTrack and BoT-SORT and to
`det_thresh` for OC-SORT and Deep OC-SORT. `output_path` defaults to
`runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Returns a metrics dictionary whose keys depend on the task; detection returns
`metrics/precision`, `metrics/recall`, `metrics/mAP50` and
`metrics/mAP50-95`. `imgsz` accepts a square int or a `(height, width)` tuple
and defaults to the model's native input size. `plots` is an alias for
`save_plots`. `allow_download_scripts` gates the embedded Python that a
dataset YAML may carry in its `download` field.

`faster_coco_eval` is accepted through `**kwargs` and defaults to `True`,
falling back to pycocotools when the package is not installed. The backend
that ran is reported on `model.last_eval_backend`.

Augmented validation raises for the `obb` and `pose` tasks.

## train

`train` is defined per family, so its arguments differ. Two behaviors are
shared, because the base class wraps every family's `train`:

- `cfg=` takes a YAML path whose keys are merged into the call. Explicit
  keyword arguments win over the file.
- `pretrained=False` on a family in coverage group `g0` or `g1` reinitializes
  the model from scratch before training, and cannot be combined with
  `resume=True`.

Which augmentation knobs a family actually honors is a per-family question;
see the [augmentation matrix](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Returns the path to the written artifact. `format` is resolved through the
exporter registry, where `engine` is an alias for `tensorrt` and `litert` is
an alias for `tflite`. Arguments shared by every exporter:

| Argument | Default | Meaning |
|---|---|---|
| `output_path` | `None` | Output file path; generated under `weights/` when omitted |
| `imgsz` | `None` | `(height, width)` tuple or a single int; defaults to the native size |
| `opset` | `None` | ONNX opset version |
| `simplify` | `True` | Run ONNX graph simplification |
| `dynamic` | `True` | Enable dynamic axes |
| `half` | `False` | FP16 precision |
| `int8` | `False` | INT8 precision |
| `batch` | `1` | Batch size baked into the artifact |
| `device` | `None` | Device to trace on |
| `data` | `None` | data.yaml for INT8 calibration |
| `fraction` | `1.0` | Fraction of the calibration dataset to use |
| `allow_download_scripts` | `False` | Allow embedded Python in dataset YAML downloads |
| `verbose` | `False` | Verbose exporter logging |

Blocked combinations raise `NotImplementedError` in preflight, before tracing.
Coverage and its rules are on the [export matrix](/docs/reference/export-matrix)
page. When live LoRA adapters are present they are folded into dense weights,
and that merge happens only after every request rejection.

## save

```python
model.save(path) -> str
```

Writes a schema v1.0 LibreYOLO checkpoint: the state dict plus the metadata
described in the [checkpoint schema](/docs/reference/checkpoint-schema).
A quantized model additionally carries its `quant` manifest, so
`LibreYOLO(path)` restores the quantized structure and scales.

## quantize, quant_info and dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Quantizes in place and returns the model. `recipe` is one of the casts
`fp16` and `bf16`, the Conv and Linear recipes `int8` and `fp8`, or the
Linear-only recipes `w4a16`, `w4a8`, `nvfp4`, `mxfp4` and `int2`, which
transformer families such as RF-DETR support. `int2` requires QAT.
`calib` takes a data.yaml path or a built-in dataset name and reads images
forward-only; labels are never read. Pass `calib=None` to skip calibration.
`algorithm` is `"minmax"`, `"percentile"` or `"auto"`.

`model.quant_info()` returns the quantization state summary, or `None` for a
float model. `model.dequantize()` restores float modules in place while
keeping the quantization-trained master weights, which is the bridge from QAT
to `export(format="onnx", int8=True, data=...)`.

## info and layers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` returns a JSON-friendly dictionary and logs a human-readable summary
when `verbose` is true. `get_available_layer_names` lists the layers a
distillation or feature-extraction config can name.

## CUDA graphs

Available on families whose `SUPPORTS_CUDA_GRAPH` class attribute is true.
Replay is bit-identical to eager execution.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # context manager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

A captured graph is valid only for the exact shape it was captured at, so
`batch` and `imgsz` must match the later `predict` call. `capture_graph` moves
the capture cost off the first request. `mode` accepts `True` or `"on"` to
capture on first use, `"auto"` to wait until a shape repeats, and `False` for
a no-op. `capture_graph` raises `NotImplementedError` when the family has not
opted in and `CudaGraphUnavailable` when capture fails.

## Device and dtype

`Results` objects carry `.to()`, `.cpu()`, `.cuda()` and `.numpy()`; see
[Results types](/docs/reference/results-types). The model itself is moved by
passing `device=` to `predict`, or at construction time.
