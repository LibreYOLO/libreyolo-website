---
title: libreyolo export
seo_title: "libreyolo export command reference"
description: "Export a checkpoint to a deployment format from the command line: arguments with their defaults, where the artifact lands, and the combinations that are refused."
lead: "Converts one checkpoint into one deployment format and writes the artifact under weights/. The format decides which of the arguments below apply."
keywords: [libreyolo export cli, libreyolo export command, yolo onnx export cli, tensorrt export command, libreyolo export arguments]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo export
    mono: true
  - label: Required
    value: model
    mono: true
  - label: Output
    value: "weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>"
    mono: true
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        # Writes weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS inside the graph
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Run the artifact
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640

        # The factory routes on the file suffix, so the export loads like a checkpoint.
        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
---

## Synopsis

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `format=onnx` and
`--format onnx` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `model` | | Model weights `.pt`. Required |
| `format` | `onnx` | Export format: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | RKNN target platform, currently `rk3588` only. Rejected with any other format |
| `imgsz` | | Input image size: `640` or `480x640` (HxW). `480,640` is also accepted. The model's own size when unset |
| `batch` | `1` | Export batch size |
| `half` | `false` | FP16 precision |
| `int8` | `false` | INT8 quantization |
| `dynamic` | `false` | Dynamic input shapes (ONNX) |
| `simplify` | `true` | ONNX graph simplification |
| `nms` | `false` | Embed NMS in the model. ONNX and CoreML only |
| `conf` | `0.25` | Confidence threshold for embedded NMS |
| `iou` | `0.45` | IoU threshold for embedded NMS |
| `max_det` | `300` | Maximum detections for ONNX embedded NMS |
| `opset` | | ONNX opset version. Chosen automatically when unset |
| `data` | | Calibration data for INT8 |
| `fraction` | `1.0` | Fraction of calibration data to use |
| `device` | `auto` | Device for tracing |
| `allow_download_scripts` | `false` | Allow embedded Python in dataset YAML download blocks |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `false` | Verbose export logging |
| `verify` | `false` | Run the RKNN Toolkit2 PC simulator and compare against ONNX Runtime. RKNN only |
| `help_json` | `false` | Dump command schema as JSON and exit |

`engine` is an alias for `tensorrt` and `litert` an alias for `tflite`. Both
resolve to the canonical name before anything is written, so the JSON output and
the log line always report `tensorrt` or `tflite`.

## Examples

<code-tabs name="examples" />

## Notes

### Where the file lands

The command takes no output path. The artifact is written to `weights/`, named
after the source checkpoint's stem plus the format's suffix, with `_fp16` or
`_int8` inserted when one of those precisions was requested. `LibreYOLO9s.pt`
exported to ONNX at FP16 becomes `weights/LibreYOLO9s_fp16.onnx`. The JSON
result carries the resolved `output_path`, the file size in MB, and the input
shape as `[batch, 3, height, width]`.

### Combinations that are refused

`nms=true` is accepted for ONNX and CoreML and refused for every other format
with `nms_unsupported_format`. On ONNX it forces `dynamic` off, since the
embedded graph is fixed at batch 1, and says so on stderr. On CoreML it takes
`conf` and `iou` but not `max_det`, so a non-default `max_det` alongside
`format=coreml nms=true` exits with `config_unsupported`.

`half=true` together with `int8=true` is not an error. INT8 wins, `half` is
dropped, and a warning goes to stderr.

`name` and `verify` are RKNN options today. Passing either with another format
exits with `config_unsupported` rather than being ignored.

### Which formats a family supports

Support is per family and per task, not global. `libreyolo formats
family=<family> task=<task>` prints the tier for each format for that
combination, with the reason and any constraint attached. See
[`libreyolo formats`](/docs/cli/utilities) for the arguments.

Some formats need an optional install and some need a toolchain. A missing
Python dependency exits with `export_dep_missing`; a precision the format
cannot produce exits with `format_precision_unsupported`.

### Running what you exported

Exported artifacts load through the same model factory as checkpoints, keyed on
the file suffix, so `libreyolo predict model=weights/LibreYOLO9s.onnx` works
without any further conversion. Three prediction options are the exception and
are refused on runtime backends: `tiling`, `overlap_ratio` and
`output_file_format`.

Two deployment targets have pages of their own:
[NVIDIA DeepStream](/docs/export/deepstream) and
[NVIDIA Jetson](/docs/export/jetson).

### Output and exit codes

stdout carries the result; progress goes to stderr. The exit code is `0` on
success, `2` for a usage or configuration error, `4` when the model cannot be
loaded, `5` for an unknown format, a missing export dependency, an unsupported
precision or a refused embedded-NMS request, and `1` for other runtime
failures.

Related: [`libreyolo quantize`](/docs/cli/quantize), which stays in PyTorch and
writes a checkpoint rather than a deployment artifact.
