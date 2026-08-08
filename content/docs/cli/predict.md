---
title: libreyolo predict
seo_title: "libreyolo predict command reference"
description: "Run inference from the command line: every argument, its default read from the CLI definition, and the flags that change what lands on stdout."
lead: "Runs a loaded model over one source and prints the predictions. The source may be an image, a directory, a video, a URL or a live stream; the model may be a checkpoint or an exported artifact."
keywords: [libreyolo predict cli, libreyolo inference command, yolo cli prediction, libreyolo predict arguments, libreyolo json output]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo predict
    mono: true
  - label: Required
    value: source
    mono: true
  - label: Output
    value: "Predictions on stdout. With save=true, annotated files under runs/detect/predict"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Save annotated images
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Filtered classes, JSON on stdout
      language: bash
      code: |
        # class 0 is person in the COCO class list the checkpoint ships with.
        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50 \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
---

## Synopsis

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Arguments are `key=value` pairs. The same command also accepts POSIX form, so
`conf=0.4` and `--conf 0.4` are interchangeable, and a boolean written
`save=true` becomes `--save`. Names with an underscore accept either spelling:
`max_det=50` and `--max-det 50` reach the same option.

`libreyolo detect predict ...` is accepted and behaves identically; the task
word is stripped before parsing.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `source` | | Image path, directory, or URL. Required |
| `model` | `yolox-s` | Model name or path |
| `conf` | `0.25` | Confidence threshold |
| `iou` | `0.45` | NMS IoU threshold |
| `imgsz` | | Input image size: `640` (square) or `480x640` (HxW). The model's own input size when unset |
| `classes` | | Filter by class IDs, e.g. `[0,2,5]`. A bare integer is accepted |
| `max_det` | `300` | Max detections per image |
| `half` | `false` | FP16 inference (CUDA only, requires model support) |
| `save` | `false` | Save annotated images |
| `batch` | `1` | Images per forward pass for directory sources. Above 1 runs true batched inference on models that support it |
| `stream` | `false` | Yield results incrementally. Turned on automatically for webcams and live streams |
| `stream_buffer` | `false` | Buffer every live frame instead of keeping only the newest |
| `vid_stride` | `1` | Process every N-th video or live frame |
| `show` | `false` | Display video and live results; `q` stops |
| `tiling` | `false` | Tiled inference for large images |
| `overlap_ratio` | `0.2` | Tile overlap ratio |
| `output_path` | | Explicit output path. Otherwise `project/name` when `save=true` |
| `color_format` | `auto` | Input color: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Output format: `jpg`, `png`, `webp` |
| `device` | `auto` | Device: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Face detector model (path or CLI name). Required for gaze models |
| `gallery` | | Face gallery `.npz` from `libreyolo enroll` to identify faces against. Face-embedding models only |
| `gallery_threshold` | `0.4` | Cosine threshold for a gallery identity match |
| `project` | `runs/detect` | Output directory root |
| `name` | `predict` | Experiment name |
| `exist_ok` | `false` | Reuse existing output directory |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `false` | Verbose stderr output |
| `help_json` | `false` | Dump command schema as JSON and exit |

## Examples

<code-tabs name="examples" />

## Notes

An exported artifact loads the same way a checkpoint does, so
`model=weights/LibreYOLO9s.onnx` and `model=weights/LibreYOLO9s.engine` are
valid values for `model`. Three options are refused on those runtimes rather
than ignored: `tiling`, `overlap_ratio` and `output_file_format` exit with
`config_unsupported` when a runtime backend cannot honor them.

`half` goes the other way. Exported runtimes receive it and run in FP16; native
PyTorch inference logs that it was ignored and continues in FP32.

Gaze models are two stage and have no detector of their own, so
`face_detector` is required for them. `gallery` applies only to models whose
task is `embed`; passing it to anything else exits with `config_unsupported`.

stdout carries results and nothing else; progress, warnings and errors go to
stderr. `json=true` prints one JSON object per invocation, or one per frame
when streaming, each carrying `schema_version`. `quiet=true` silences stderr.
Both together give a machine reader a clean stdout stream.

The exit code is `0` on success, `2` for a usage or configuration error, `3`
when the source cannot be found, `4` when the model cannot be loaded, and `1`
for other runtime failures.

`help_json=true` prints the command's parameters, types, defaults and flags as
JSON without running anything, which is the reliable way to read this table
back from an installed version.

Related: [`libreyolo val`](/docs/cli/val) for measured metrics on a dataset,
[`libreyolo export`](/docs/cli/export) to produce the runtime artifacts named
above.
