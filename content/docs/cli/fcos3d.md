---
title: "libreyolo fcos3d"
seo_title: "libreyolo fcos3d: arguments and examples"
description: "Command-line reference for fcos3d 3D detection, including calibration, runtime options and JSON output."
lead: "Run fcos3d 3D detection from the command line."
keywords: [fcos3d, CLI, LibreYOLO, 3D detection]
last_verified: "1.6.0"
---

## Synopsis

```bash
libreyolo fcos3d source=IMAGE intrinsics=camera.npy [OPTIONS]
```

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | `required` | Image path or directory |
| `model` | `upstream default checkpoint` | Local checkpoint or the mirrored default filename |
| `intrinsics` | `required` | Original-image 3x3 camera calibration .npy file |
| `device` | `auto` | auto, cpu, or a CUDA device |
| `conf` | `None` | Joint confidence threshold |
| `iou` | `None` | Rotated bird's-eye-view NMS threshold |
| `max-det` | `200` | Maximum number of detections per image |
| `save` | `False` | Save projected cuboids |
| `output-path` | `None` | Single-image output filename |
| `json-output` | `False` | JSON output to stdout |
| `quiet` | `False` | Suppress stderr |
| `help-json` | `False` | Dump command schema as JSON |

## Examples

Inspect the installed command schema:

```bash
libreyolo fcos3d --help-json
```

Predict on the shipped sample image. For commands requiring calibration,
`camera.npy` must contain its measured original-image 3x3 intrinsic matrix.
Install the model's runtime before prediction.

```bash
libreyolo fcos3d source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg intrinsics=camera.npy
```

Request machine-readable results:

```bash
libreyolo fcos3d source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg intrinsics=camera.npy --json --quiet
```

## Notes

Configuration, runtime-loading and I/O errors exit through the shared CLI
error handler. `--json` writes structured results to stdout; diagnostics use
stderr. `--quiet` suppresses stderr. See [fcos3d](/docs/models/fcos3d)
for installation, calibration and model constraints.
