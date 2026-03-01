# LibreYOLO Documentation

---

# Introduction

LibreYOLO is an MIT-licensed object detection library that provides a unified Python API across three architectures: **YOLOX**, **YOLOv9**, and **RF-DETR**. One interface for prediction, training, validation, and export — regardless of which model family you use.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
results = model("image.jpg", conf=0.25, save=True)
print(results.boxes.xyxy)
```

### Key features

- Unified API across YOLOX, YOLOv9, and RF-DETR
- Auto-detection of model architecture, size, and class count from weights
- Tiled inference for large/high-resolution images
- ONNX, TorchScript, TensorRT, and OpenVINO export with embedded metadata
- ONNX Runtime, TensorRT, and OpenVINO inference backends
- COCO-compatible validation with mAP metrics
- Accepts any image format: file paths, URLs, PIL, NumPy, PyTorch tensors, raw bytes

---

# Installation

### Requirements

- Python 3.10+
- PyTorch 1.7+

### From PyPI

```bash
pip install libreyolo
```

### From source

```bash
git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .
```

### Optional dependencies

```bash
# ONNX export and inference
pip install libreyolo[onnx]
# or: pip install onnx onnxsim onnxscript onnxruntime

# RF-DETR support
pip install libreyolo[rfdetr]
# or: pip install rfdetr transformers timm supervision

# TensorRT export and inference (NVIDIA GPU)
# Note: TensorRT itself requires manual installation (depends on CUDA version)
pip install libreyolo[tensorrt]

# OpenVINO export and inference (Intel CPU/GPU/VPU)
pip install libreyolo[openvino]
```

If using `uv`:

```bash
uv sync --extra onnx
uv sync --extra rfdetr
uv sync --extra tensorrt
uv sync --extra openvino
```

---

# Quickstart

### Load a model and run inference

```python
from libreyolo import LibreYOLO

# Auto-detects architecture and size from the weights file
model = LibreYOLO("LibreYOLOXs.pt")

# Run on a single image
result = model("photo.jpg")

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)   # bounding boxes (N, 4)
print(result.boxes.conf)   # confidence scores (N,)
print(result.boxes.cls)    # class IDs (N,)
```

### Save annotated output

```python
result = model("photo.jpg", save=True)
# Saved to runs/detections/photo_LibreYOLOX_s_<timestamp>.jpg
```

### Process a directory

```python
results = model("images/", save=True, batch=4)
for r in results:
    print(f"{r.path}: {len(r)} detections")
```

---

# Available Models

### YOLOX

| Size | Code | Input size | Use case |
|------|------|-----------|----------|
| Nano | `"n"` | 416 | Edge devices, mobile |
| Tiny | `"t"` | 416 | Edge devices, faster |
| Small | `"s"` | 640 | Balanced speed/accuracy |
| Medium | `"m"` | 640 | Higher accuracy |
| Large | `"l"` | 640 | High accuracy |
| XLarge | `"x"` | 640 | Maximum accuracy |

```python
from libreyolo import LibreYOLOX
model = LibreYOLOX("LibreYOLOXs.pt", size="s")
```

### YOLOv9

| Size | Code | Input size | Use case |
|------|------|-----------|----------|
| Tiny | `"t"` | 640 | Fast inference |
| Small | `"s"` | 640 | Balanced |
| Medium | `"m"` | 640 | Higher accuracy |
| Compact | `"c"` | 640 | Best accuracy |

```python
from libreyolo import LibreYOLO9
model = LibreYOLO9("LibreYOLO9c.pt", size="c")
```

### RF-DETR

| Size | Code | Input size | Use case |
|------|------|-----------|----------|
| Nano | `"n"` | 384 | Edge |
| Small | `"s"` | 512 | Balanced |
| Medium | `"m"` | 576 | Higher accuracy |
| Large | `"l"` | 704 | Maximum accuracy |

```python
from libreyolo import LibreYOLORFDETR
model = LibreYOLORFDETR(size="s")
```

### Factory function (recommended)

The `LibreYOLO()` factory auto-detects everything from the weights file:

```python
from libreyolo import LibreYOLO

# Auto-detects: YOLOX, size=s, 80 classes
model = LibreYOLO("LibreYOLOXs.pt")

# Auto-detects: YOLOv9, size=c, 80 classes
model = LibreYOLO("LibreYOLO9c.pt")

# Auto-detects: RF-DETR
model = LibreYOLO("LibreRFDETRs.pth")

# ONNX models work too
model = LibreYOLO("model.onnx")

# TensorRT engines
model = LibreYOLO("model.engine")

# OpenVINO models (directory with model.xml)
model = LibreYOLO("model_openvino/")
```

If weights are not found locally, LibreYOLO attempts to download them from Hugging Face automatically.

---

# Prediction

### Basic prediction

```python
result = model("image.jpg")
```

### All prediction parameters

```python
result = model(
    "image.jpg",
    conf=0.25,            # confidence threshold (default: 0.25)
    iou=0.45,             # NMS IoU threshold (default: 0.45)
    imgsz=640,            # input size override (default: model's native)
    classes=[0, 2, 5],    # filter to specific class IDs (default: all)
    max_det=300,          # max detections per image (default: 300)
    save=True,            # save annotated image (default: False)
    output_path="out/",   # where to save (default: runs/detections/)
    color_format="auto",  # "auto", "rgb", or "bgr" (default: "auto")
    output_file_format="png",  # output format: "jpg", "png", "webp"
)
```

`model.predict(...)` is an alias for `model(...)`.

### Supported input formats

LibreYOLO accepts images in any of these formats:

```python
# File path (string or pathlib.Path)
result = model("photo.jpg")
result = model(Path("photo.jpg"))

# URL
result = model("https://example.com/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("photo.jpg")
result = model(img)

# NumPy array (HWC or CHW, RGB or BGR, uint8 or float32)
import numpy as np
arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = model(arr)

# OpenCV (BGR) — specify color_format
import cv2
frame = cv2.imread("photo.jpg")
result = model(frame, color_format="bgr")

# PyTorch tensor (CHW or NCHW)
import torch
tensor = torch.randn(3, 640, 640)
result = model(tensor)

# Raw bytes
with open("photo.jpg", "rb") as f:
    result = model(f.read())

# Directory of images
results = model("images/", batch=4)
```

### Working with results

Every prediction returns a `Results` object (or a list of them for directories):

```python
result = model("image.jpg")

# Number of detections
len(result)  # e.g., 5

# Bounding boxes in xyxy format (x1, y1, x2, y2)
result.boxes.xyxy        # tensor of shape (N, 4)

# Bounding boxes in xywh format (center_x, center_y, width, height)
result.boxes.xywh        # tensor of shape (N, 4)

# Confidence scores
result.boxes.conf        # tensor of shape (N,)

# Class IDs
result.boxes.cls         # tensor of shape (N,)

# Combined data: [x1, y1, x2, y2, conf, cls]
result.boxes.data        # tensor of shape (N, 6)

# Metadata
result.orig_shape        # (height, width) of original image
result.path              # source file path (or None)
result.names             # {0: "person", 1: "bicycle", ...}

# Move to CPU / convert to numpy
result_cpu = result.cpu()
boxes_np = result.boxes.numpy()       # Boxes object backed by numpy arrays
xyxy_np = result.boxes.xyxy.numpy()   # raw numpy array
```

### Class filtering

Filter detections to specific class IDs:

```python
# Only detect people (class 0) and cars (class 2)
result = model("image.jpg", classes=[0, 2])
```

---

# Tiled Inference

For images much larger than the model's input size (e.g., satellite imagery, drone footage), tiled inference splits the image into overlapping tiles, runs detection on each, and merges results.

```python
result = model(
    "large_aerial_image.jpg",
    tiling=True,
    overlap_ratio=0.2,   # 20% overlap between tiles (default)
    save=True,
)

# Extra metadata on tiled results
result.tiled           # True
result.num_tiles       # number of tiles used
result.saved_path      # directory containing saved results (when save=True)
result.tiles_path      # path to tiles/ subdirectory
result.grid_path       # path to grid visualization image
```

When `save=True` with tiling, LibreYOLO saves outputs in a timestamped directory (`runs/tiled_detections/<image>_<model>_<timestamp>/`):
- `final_image.jpg` — full image with all merged detections drawn
- `grid_visualization.jpg` — image showing tile grid overlay
- `tiles/` — individual tile crops
- `metadata.json` — tiling parameters and detection counts

If the image is already smaller than the model's input size, tiling is skipped automatically.

---

# Training

### YOLOX training

```python
from libreyolo import LibreYOLOX

model = LibreYOLOX(size="s")

results = model.train(
    data="coco128.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=100,
    batch=16,
    imgsz=640,

    # Optimizer
    lr0=0.01,                # initial learning rate
    optimizer="SGD",         # "SGD", "Adam", "AdamW"

    # System
    device="0",              # GPU device ("", "cpu", "cuda", "0", "0,1")
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")
```

After training, the model instance is automatically updated with the best weights.

#### Training results dict

```python
{
    "final_loss": 2.31,
    "best_mAP50": 0.682,
    "best_mAP50_95": 0.451,
    "best_epoch": 87,
    "save_dir": "runs/train/exp",
    "best_checkpoint": "runs/train/exp/weights/best.pt",
    "last_checkpoint": "runs/train/exp/weights/last.pt",
}
```

#### Resuming training

```python
model = LibreYOLOX("runs/train/exp/weights/last.pt", size="s")
results = model.train(data="coco128.yaml", resume=True)
```

#### Custom dataset YAML format

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names: ["cat", "dog", "bird"]
```

### YOLOv9 training

YOLOv9 training uses the same API as YOLOX with slightly different defaults:

```python
from libreyolo import LibreYOLO9

model = LibreYOLO9("LibreYOLO9c.pt", size="c")

results = model.train(
    data="coco128.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=300,              # default: 300 (vs 100 for YOLOX)
    batch=16,
    imgsz=640,

    # Optimizer
    lr0=0.01,
    optimizer="SGD",

    # System
    device="0",
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="v9_exp",           # default: "v9_exp"
    exist_ok=False,

    # Training features
    amp=True,
    patience=50,
    resume=False,
)
```

### RF-DETR training

RF-DETR uses a different training API that wraps the original rfdetr implementation:

```python
from libreyolo import LibreYOLORFDETR

model = LibreYOLORFDETR(size="s")

results = model.train(
    data="path/to/dataset",  # Roboflow/COCO format directory
    epochs=100,
    batch_size=4,
    lr=1e-4,
    output_dir="runs/train",
)
```

RF-DETR datasets use COCO annotation format:

```
dataset/
    train/
        _annotations.coco.json
        image1.jpg
        image2.jpg
    valid/
        _annotations.coco.json
        image1.jpg
```

---

# Validation

Run COCO-standard evaluation on a validation set:

```python
results = model.val(
    data="coco128.yaml",   # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=False,       # save predictions as COCO JSON
    plots=True,            # reserved for future visualization
    verbose=True,          # print per-class metrics
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
```

### Validation results dict

By default, LibreYOLO uses COCO evaluation (`use_coco_eval=True` in `ValidationConfig`):

```python
{
    "metrics/mAP50-95": 0.489,     # COCO primary metric (AP@[.5:.95])
    "metrics/mAP50": 0.721,        # AP@0.5 (PASCAL VOC style)
    "metrics/mAP75": 0.512,        # AP@0.75 (strict)
    "metrics/mAP_small": 0.291,
    "metrics/mAP_medium": 0.534,
    "metrics/mAP_large": 0.632,
    "metrics/AR1": 0.362,          # Average Recall (max 1 det)
    "metrics/AR10": 0.571,
    "metrics/AR100": 0.601,
    "metrics/AR_small": 0.387,
    "metrics/AR_medium": 0.655,
    "metrics/AR_large": 0.741,
}
```

---

# Export

Export models to ONNX, TorchScript, TensorRT, or OpenVINO for deployment.

### Quick export

```python
# ONNX (default)
model.export()

# TorchScript
model.export(format="torchscript")

# TensorRT with FP16
model.export(format="tensorrt", half=True)

# OpenVINO
model.export(format="openvino")
```

### All export parameters

```python
path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", or "openvino"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native)
    opset=13,                 # ONNX opset version (default: 13)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis
    half=False,               # export in FP16
    int8=False,               # export in INT8 (requires data for calibration)
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    # INT8 calibration
    data=None,                # path to data.yaml for INT8 calibration
    fraction=1.0,             # fraction of calibration dataset to use
    # TensorRT specific
    workspace=4.0,            # TensorRT workspace size in GiB
    hardware_compatibility="none",  # "none", "ampere_plus", "same_compute_capability"
    gpu_device=0,             # GPU device ID
    trt_config=None,          # TensorRT config YAML file
    verbose=False,            # enable verbose logging
)
```

### TensorRT export

```python
# TensorRT with FP16 (fastest on NVIDIA GPU)
path = model.export(format="tensorrt", half=True)

# TensorRT with INT8 quantization (requires calibration dataset)
path = model.export(
    format="tensorrt",
    int8=True,
    data="coco8.yaml",
    workspace=8.0,
)

# TensorRT with config file
path = model.export(format="tensorrt", trt_config="tensorrt.yaml")
```

### OpenVINO export

```python
# OpenVINO IR format (Intel CPU/GPU/VPU)
path = model.export(format="openvino")

# OpenVINO with FP16 compression
path = model.export(format="openvino", half=True)

# OpenVINO with INT8 quantization
path = model.export(format="openvino", int8=True, data="coco8.yaml")
```

### ONNX metadata

Exported ONNX files include embedded metadata:

| Key | Example value |
|-----|--------------|
| `libreyolo_version` | `"0.1.4"` |
| `model_family` | `"LibreYOLOX"` |
| `model_size` | `"s"` |
| `nb_classes` | `"80"` |
| `names` | `'{"0": "person", "1": "bicycle", ...}'` |
| `imgsz` | `"640"` |
| `dynamic` | `"True"` |
| `half` | `"False"` |

This metadata is automatically read back when loading the model with `LibreYOLOOnnx` (see next section).

### Using the Exporter class directly

```python
from libreyolo.export import Exporter

exporter = Exporter(model)
path = exporter("onnx", dynamic=True, simplify=True)
```

---

# Inference Backends

LibreYOLO supports three inference backends beyond PyTorch: ONNX Runtime, TensorRT, and OpenVINO. All provide the same prediction API.

## ONNX Inference

Run inference using ONNX Runtime instead of PyTorch. Useful for deployment environments without PyTorch.

```python
from libreyolo import LibreYOLOOnnx

model = LibreYOLOOnnx("model.onnx")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)
```

### Auto-metadata

If the ONNX file was exported by LibreYOLO, class names and class count are read automatically from the embedded metadata:

```python
# Export with metadata
model.export(format="onnx", output_path="model.onnx")

# Load — names and nb_classes auto-populated
onnx_model = LibreYOLOOnnx("model.onnx")
print(onnx_model.names)       # {0: "person", 1: "bicycle", ...}
print(onnx_model.nb_classes)  # 80
```

For ONNX files without metadata (e.g., exported by other tools), specify `nb_classes` manually:

```python
model = LibreYOLOOnnx("external_model.onnx", nb_classes=20)
```

### Device selection

```python
# Auto-detect (CUDA if available, else CPU)
model = LibreYOLOOnnx("model.onnx", device="auto")

# Force CPU
model = LibreYOLOOnnx("model.onnx", device="cpu")

# Force CUDA
model = LibreYOLOOnnx("model.onnx", device="cuda")
```

### Prediction parameters

`LibreYOLOOnnx` supports the same prediction API:

```python
result = model(
    "image.jpg",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    classes=[0, 2],
    max_det=300,
    save=True,
    output_path="output/",
    color_format="rgb",
)
```

## TensorRT Inference

Run inference using TensorRT for GPU-accelerated deployment on NVIDIA hardware:

```python
from libreyolo.common.tensorrt import LibreYOLOTensorRT

model = LibreYOLOTensorRT("model.engine")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)
```

Or load via the factory (auto-detects `.engine` files):

```python
from libreyolo import LibreYOLO
model = LibreYOLO("model.engine")
```

## OpenVINO Inference

Run inference using OpenVINO for Intel CPU/GPU/VPU acceleration:

```python
from libreyolo import LibreYOLOOpenVINO

model = LibreYOLOOpenVINO("model_openvino/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)
```

Or load via the factory (auto-detects OpenVINO directories):

```python
from libreyolo import LibreYOLO
model = LibreYOLO("model_openvino/")
```

---

# API Reference

## LibreYOLO (factory)

```python
LibreYOLO(
    model_path: str,
    size: str = None,           # auto-detected from weights
    reg_max: int = 16,          # YOLOv9 only
    nb_classes: int = None,     # auto-detected from weights
    device: str = "auto",
) -> LibreYOLOX | LibreYOLO9 | LibreYOLORFDETR | LibreYOLOOnnx | LibreYOLOTensorRT | LibreYOLOOpenVINO
```

Auto-detects model architecture, size, and class count from the weights file. Returns the appropriate model class. Also handles `.onnx`, `.engine`, and OpenVINO directories.

Downloads weights from Hugging Face if not found locally.

## LibreYOLOX

```python
LibreYOLOX(
    model_path=None,            # str, dict, or None
    size: str = "s",            # "n", "t", "s", "m", "l", "x"
    nb_classes: int = 80,
    device: str = "auto",
)
```

Pass `model_path=None` to create a fresh (randomly initialized) model for training.

## LibreYOLO9

```python
LibreYOLO9(
    model_path,                 # str or dict (required)
    size: str,                  # "t", "s", "m", "c" (required)
    reg_max: int = 16,
    nb_classes: int = 80,
    device: str = "auto",
)
```

## LibreYOLORFDETR

```python
LibreYOLORFDETR(
    model_path: str = None,     # None = use default pretrained
    size: str = "s",            # "n", "s", "m", "l"
    nb_classes: int = 80,
    device: str = "auto",
)
```

## Prediction (all models)

```python
model(
    source,                     # image input (see supported formats)
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    batch: int = 1,
    output_path: str = None,
    color_format: str = "auto",
    tiling: bool = False,
    overlap_ratio: float = 0.2,
    output_file_format: str = None,
) -> Results | list[Results]
```

## Results

```python
result = Results(
    boxes: Boxes,
    orig_shape: tuple[int, int],  # (height, width)
    path: str | None,
    names: dict[int, str],
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU
```

## Boxes

```python
boxes = Boxes(boxes, conf, cls)

boxes.xyxy           # (N, 4) tensor — x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor — cx, cy, w, h
boxes.conf           # (N,) tensor — confidence scores
boxes.cls            # (N,) tensor — class IDs
boxes.data           # (N, 6) tensor — [xyxy, conf, cls]

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays
```

## model.export()

```python
model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino"
    *,
    output_path: str = None,
    imgsz: int = None,
    opset: int = 13,
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    int8: bool = False,
    batch: int = 1,
    device: str = None,
    data: str = None,           # calibration dataset for INT8
    fraction: float = 1.0,      # fraction of calibration dataset
    workspace: float = 4.0,     # TensorRT workspace (GiB)
    hardware_compatibility: str = "none",  # TensorRT HW compat level
    gpu_device: int = 0,        # GPU device ID
    trt_config: str = None,     # TensorRT config YAML file
    verbose: bool = False,
) -> str                        # path to exported file
```

## Exporter

```python
from libreyolo.export import Exporter

exporter = Exporter(model)
path = exporter(
    format,                     # same parameters as model.export()
    **kwargs,
)

Exporter.FORMATS               # dict of supported formats
# {"onnx": {"suffix": ".onnx", "requires": None},
#  "torchscript": {"suffix": ".torchscript", "requires": None},
#  "tensorrt": {"suffix": ".engine", "requires": "onnx"},
#  "openvino": {"suffix": "_openvino", "requires": "onnx"}}
```

## model.val()

```python
model.val(
    data: str = None,           # path to data.yaml
    batch: int = 16,
    imgsz: int = None,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = None,
    split: str = "val",
    save_json: bool = False,
    plots: bool = True,
    verbose: bool = True,
) -> dict
```

Returns (with default COCO evaluation):

```python
{
    "metrics/mAP50-95": float,     # COCO primary metric
    "metrics/mAP50": float,
    "metrics/mAP75": float,
    "metrics/mAP_small": float,
    "metrics/mAP_medium": float,
    "metrics/mAP_large": float,
    "metrics/AR1": float,
    "metrics/AR10": float,
    "metrics/AR100": float,
    "metrics/AR_small": float,
    "metrics/AR_medium": float,
    "metrics/AR_large": float,
}
```

## model.train() (YOLOX)

```python
model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict
```

Returns:

```python
{
    "final_loss": float,
    "best_mAP50": float,
    "best_mAP50_95": float,
    "best_epoch": int,
    "save_dir": str,
    "best_checkpoint": str,
    "last_checkpoint": str,
}
```

## model.train() (YOLOv9)

```python
model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "v9_exp",
    exist_ok: bool = False,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict
```

Returns the same dict as YOLOX training.

## model.train() (RF-DETR)

```python
model.train(
    data: str,                  # path to dataset directory
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs,                   # additional args passed to rfdetr trainer
) -> dict
```

Returns:

```python
{
    "output_dir": str,          # path to training output directory
    "model": object,            # trained model instance
}
```

## LibreYOLOOnnx

```python
LibreYOLOOnnx(
    onnx_path: str,
    nb_classes: int = 80,       # auto-read from metadata if available
    device: str = "auto",
)
```

Supports the same prediction API as PyTorch models (except tiling).

## ValidationConfig

```python
from libreyolo import ValidationConfig

config = ValidationConfig(
    data="coco128.yaml",
    data_dir=None,              # direct dataset path (alternative to data)
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    use_coco_eval=True,         # use pycocotools evaluation
    split="val",
    device="auto",
    save_json=False,
    plots=True,
    verbose=True,
    num_workers=4,
    half=False,
)

# Load/save YAML
config = ValidationConfig.from_yaml("config.yaml")
config.to_yaml("config.yaml")
```

---

# Architecture Guide

This section is for contributors who want to understand the codebase internals.

### Base class design

All model families inherit from `LibreYOLOBase` (in `libreyolo/common/base_model.py`). Subclasses implement these abstract methods:

| Method | Purpose |
|--------|---------|
| `_init_model()` | Build and return the `nn.Module` |
| `_get_valid_sizes()` | Return list of valid size codes |
| `_get_model_name()` | Return model name string |
| `_get_input_size()` | Return default input resolution |
| `_preprocess()` | Image to tensor conversion |
| `_forward()` | Model forward pass |
| `_postprocess()` | Raw output to detection dicts |
| `_get_available_layers()` | Map of named layers |

The base class provides the shared pipeline: `__call__` / `predict`, `export`, `val`, tiled inference, results wrapping, and saving.

### Package structure

```
libreyolo/
    __init__.py          # Public API exports
    factory.py           # LibreYOLO() auto-detection factory
    common/
        base_model.py    # LibreYOLOBase abstract class
        onnx.py          # LibreYOLOOnnx runtime backend
        tensorrt.py      # LibreYOLOTensorRT runtime backend
        openvino.py      # LibreYOLOOpenVINO runtime backend
        results.py       # Results and Boxes classes
        image_loader.py  # Unified image loading
        utils.py         # NMS, drawing, preprocessing
    export/
        exporter.py      # Unified Exporter class
        onnx.py          # ONNX export implementation
        torchscript.py   # TorchScript export implementation
        tensorrt.py      # TensorRT export implementation
        openvino.py      # OpenVINO export implementation
        calibration.py   # INT8 calibration utilities
        config.py        # Export configuration
    yolox/
        model.py         # LibreYOLOX
        nn.py            # YOLOX network architecture
        utils.py         # YOLOX-specific pre/postprocessing
    v9/
        model.py         # LibreYOLO9
        nn.py            # YOLOv9 network architecture
        utils.py         # v9-specific pre/postprocessing
        config.py        # YOLOv9 configuration
        loss.py          # YOLOv9 loss functions
        trainer.py       # YOLOv9 training
    rfdetr/
        model.py         # LibreYOLORFDETR
        nn.py            # RF-DETR network + configs
        utils.py         # RF-DETR postprocessing
        train.py         # RF-DETR training wrapper
    training/
        config.py        # YOLOXTrainConfig
        trainer.py       # YOLOXTrainer
        dataset.py       # Training dataset
        augment.py       # Mosaic, mixup, etc.
        loss.py          # YOLOX loss functions
        scheduler.py     # LR schedulers
        ema.py           # Exponential moving average
    validation/
        config.py        # ValidationConfig
        detection_validator.py  # DetectionValidator
        metrics.py       # DetMetrics, mAP computation
        base.py          # BaseValidator
        preprocessors.py # Per-model val preprocessing
        coco_evaluator.py # COCO evaluation helper
        utils.py         # Validation utilities
    data/
        utils.py         # Dataset loading, YAML parsing
        yolo_coco_api.py # YOLO-to-COCO annotation bridge
    cfg/
        datasets/        # Built-in dataset YAML configs
```

### Adding a new model family

1. Create `libreyolo/newmodel/model.py` with a class inheriting `LibreYOLOBase`
2. Implement all abstract methods
3. Create `libreyolo/newmodel/nn.py` with the actual network architecture
4. Add detection logic to `factory.py` for auto-detection from weights
5. Export the class from `libreyolo/__init__.py`
6. (Optional) Override `_get_val_preprocessor()` if the model needs non-standard validation preprocessing

### Export architecture

The `Exporter` class (in `libreyolo/export/exporter.py`) is format-agnostic. It uses a `FORMATS` dict for dispatch:

```python
FORMATS = {
    "onnx":        {"suffix": ".onnx",        "requires": None},
    "torchscript": {"suffix": ".torchscript", "requires": None},
    "tensorrt":    {"suffix": ".engine",      "requires": "onnx"},
    "openvino":    {"suffix": "_openvino",    "requires": "onnx"},
}
```

To add a new export format, add an entry to `FORMATS` with a `suffix` and optional `requires` dependency, then implement the corresponding export function.

---

# Dataset Format

LibreYOLO uses YOLO-format datasets configured via YAML files.

### data.yaml structure

```yaml
path: /absolute/path/to/dataset   # dataset root
train: images/train               # relative to path
val: images/val                   # relative to path
test: images/test                 # optional

nc: 80                            # number of classes
names: [                          # class names
  "person", "bicycle", "car", "motorcycle", "airplane",
  "bus", "train", "truck", "boat", "traffic light",
  # ...
]
```

### Directory layout

```
dataset/
    images/
        train/
            img001.jpg
            img002.jpg
        val/
            img003.jpg
    labels/
        train/
            img001.txt
            img002.txt
        val/
            img003.txt
```

### Label format

One text file per image. Each line is one object:

```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized to [0, 1] relative to image dimensions.

Example (`img001.txt`):
```
0 0.5 0.4 0.3 0.6
2 0.1 0.2 0.05 0.1
```

### Built-in datasets

LibreYOLO includes configs for common datasets that auto-download:

```python
# These download automatically on first use
results = model.val(data="coco8.yaml")            # 8 images
results = model.train(data="coco128.yaml", epochs=10)  # 128 images
results = model.val(data="coco5000.yaml")          # 5000 images
results = model.val(data="coco.yaml")              # full COCO dataset
results = model.val(data="coco-val-only.yaml")     # validation split only
```

All available configs are in `libreyolo/cfg/datasets/`.

### RF-DETR dataset format

RF-DETR uses COCO-format annotations (JSON) instead of YOLO text labels:

```
dataset/
    train/
        _annotations.coco.json
        image1.jpg
    valid/
        _annotations.coco.json
        image1.jpg
```
