---
title: "libreyolo detany3d"
seo_title: "libreyolo detany3d: arguments and examples"
description: "Command-line reference for detany3d 3D detection, including prompts, runtime options and JSON output."
lead: "Run detany3d 3D detection from the command line."
keywords: [detany3d, CLI, LibreYOLO, 3D detection]
last_verified: "1.6.0"
---

## Synopsis

```bash
libreyolo detany3d source=IMAGE [OPTIONS]
```

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | `required` | Image path or directory |
| `model` | `upstream default checkpoint` | Local checkpoint or the mirrored default filename |
| `runtime-path` | `None` | DetAny3D checkout; defaults to DETANY3D_PATH |
| `runtime-python` | `None` | Interpreter for the separate runtime environment |
| `text` | `None` | JSON category list or a text caption |
| `bboxes` | `None` | JSON original-pixel xyxy box list |
| `points` | `None` | JSON positive points, optionally grouped by object |
| `conf` | `None` | Text detector box-confidence threshold |
| `text-threshold` | `None` | Text token-confidence threshold |
| `grounding-checkpoint` | `None` | Optional GroundingDINO Swin-B checkpoint path |
| `grounding-config` | `None` | Optional GroundingDINO Swin-B config path |
| `device` | `auto` | auto, cpu, or CUDA device |
| `save` | `False` | Save projected cuboids |
| `output-path` | `None` | Single-image output filename |
| `json` | `false` | JSON output to stdout |
| `quiet` | `False` | Suppress stderr |
| `help-json` | `false` | Flag: dump the command schema as JSON and exit |

## Examples

Inspect the installed command schema:

```bash
libreyolo detany3d --help-json
```

Predict on the shipped sample image. DetAny3D estimates the camera
intrinsics itself, so no calibration file is needed.
Install the model's runtime before prediction.

```bash
libreyolo detany3d source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg text='["person"]'
```

Request machine-readable results:

```bash
libreyolo detany3d source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg text='["person"]' --json --quiet
```

## Notes

Configuration, runtime-loading and I/O errors exit through the shared CLI
error handler. `--json` writes structured results to stdout; diagnostics use
stderr. `--quiet` suppresses stderr. See [detany3d](/docs/models/detany3d)
for installation, calibration and model constraints.
