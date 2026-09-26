---
title: "libreyolo 3dmood"
seo_title: "libreyolo 3dmood: arguments and examples"
description: "Command-line reference for 3dmood 3D detection, including calibration, runtime options and JSON output."
lead: "Run 3dmood 3D detection from the command line."
keywords: [3dmood, CLI, LibreYOLO, 3D detection]
last_verified: "1.6.0"
---

## Synopsis

```bash
libreyolo 3dmood source=IMAGE intrinsics=camera.npy text='["CLASS"]' [OPTIONS]
```

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | `required` | Image path or directory |
| `model` | `upstream default checkpoint` | Local checkpoint path |
| `size` | `None` | Model size: t or b; inferred from official filenames |
| `intrinsics` | `required` | Original-image 3x3 calibration .npy file |
| `text` | `required` | JSON category list, e.g. ["car","person"] |
| `device` | `auto` | auto, cpu, mps, or a CUDA device |
| `runtime-path` | `None` | Optional upstream 3D-MOOD checkout |
| `runtime-python` | `None` | Python interpreter for the upstream runtime |
| `conf` | `None` | Detection confidence threshold |
| `iou` | `None` | Class-agnostic NMS IoU threshold |
| `max-det` | `None` | Maximum detections per image |
| `save` | `False` | Save projected cuboids |
| `output-path` | `None` | Single-image output filename |
| `json` | `false` | JSON output to stdout |
| `quiet` | `False` | Suppress stderr |
| `help-json` | `False` | Dump command schema as JSON |

## Examples

Inspect the installed command schema:

```bash
libreyolo 3dmood --help-json
```

Predict on the shipped sample image. For commands requiring calibration,
`camera.npy` must contain its measured original-image 3x3 intrinsic matrix.
Install the model's runtime before prediction.

```bash
libreyolo 3dmood source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg intrinsics=camera.npy text='["person"]'
```

Request machine-readable results:

```bash
libreyolo 3dmood source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg intrinsics=camera.npy text='["person"]' --json --quiet
```

## Notes

Configuration, runtime-loading and I/O errors exit through the shared CLI
error handler. `--json` writes structured results to stdout; diagnostics use
stderr. `--quiet` suppresses stderr. See [3d-mood](/docs/models/3d-mood)
for installation, calibration and model constraints.
