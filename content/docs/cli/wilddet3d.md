---
title: "libreyolo wilddet3d"
seo_title: "libreyolo wilddet3d: arguments and examples"
description: "Command-line reference for wilddet3d 3D detection, including calibration, runtime options and JSON output."
lead: "Run wilddet3d 3D detection from the command line."
keywords: [wilddet3d, CLI, LibreYOLO, 3D detection]
last_verified: "1.6.0"
---

## Synopsis

```bash
libreyolo wilddet3d source=IMAGE intrinsics=camera.npy [OPTIONS]
```

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | `required` | Image path or directory |
| `model` | `upstream default checkpoint` | Local checkpoint or the mirrored default filename |
| `intrinsics` | `required` | Original-image 3x3 calibration .npy file |
| `text` | `None` | JSON category list, e.g. ["car","person"] |
| `bboxes` | `None` | JSON original-pixel xyxy box list |
| `points` | `None` | JSON xy points, optionally grouped |
| `labels` | `None` | JSON binary labels matching points |
| `prompt-mode` | `geometric` | geometric or visual |
| `depth` | `None` | Original-resolution depth .npy in metres |
| `device` | `auto` | auto, cpu, or a CUDA device |
| `runtime-path` | `None` | Optional upstream runtime checkout |
| `runtime-python` | `None` | Python interpreter for the Mac/CPU runtime |
| `conf` | `None` | Combined ranking-score threshold |
| `conf3d` | `None` | 3D confidence threshold |
| `iou` | `None` | 2D NMS IoU threshold |
| `save` | `False` | Save projected cuboids |
| `output-path` | `None` | Single-image output filename |
| `json` | `false` | JSON output to stdout |
| `quiet` | `False` | Suppress stderr |
| `help-json` | `False` | Dump command schema as JSON |

## Examples

Inspect the installed command schema:

```bash
libreyolo wilddet3d --help-json
```

Predict on the shipped sample image. For commands requiring calibration,
`camera.npy` must contain its measured original-image 3x3 intrinsic matrix.
Install the model's runtime before prediction. Pass exactly one prompt:
`text`, `bboxes` or `points`.

```bash
libreyolo wilddet3d source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg intrinsics=camera.npy text='["person"]'
```

Request machine-readable results:

```bash
libreyolo wilddet3d source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg intrinsics=camera.npy text='["person"]' --json --quiet
```

## Notes

Configuration, runtime-loading and I/O errors exit through the shared CLI
error handler. `--json` writes structured results to stdout; diagnostics use
stderr. `--quiet` suppresses stderr. See [wilddet3d](/docs/models/wilddet3d)
for installation, calibration and model constraints.
