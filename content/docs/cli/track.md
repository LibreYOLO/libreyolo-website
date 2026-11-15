---
title: libreyolo track
seo_title: "libreyolo track command reference"
description: "Track objects across a video or an image sequence from the command line: every argument with its default, the four trackers, and the one-line-per-frame JSON output."
lead: "Runs model.track() over a video file, an image directory or an image list and prints one line per frame with that frame's track IDs. New in 1.8.0, and it works with checkpoints and exported artifacts alike."
keywords: [libreyolo track cli, libreyolo tracking command, bytetrack cli, botsort command line, multi object tracking cli, libreyolo track json]
last_verified: "1.8.0"
meta:
  - label: Command
    value: libreyolo track
    mono: true
  - label: Required
    value: source
    mono: true
  - label: Output
    value: "One line per frame on stdout. With save=true, an annotated video under runs/track/"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        libreyolo track model=LibreYOLO9s.pt source=video.mp4
    - label: Another tracker, saved video
      language: bash
      code: |
        libreyolo track model=LibreYOLO9s.pt source=video.mp4 \
          tracker=botsort classes="[0]" save=true
    - label: Exported model, JSON per frame
      language: bash
      code: |
        libreyolo track model=weights/LibreYOLO9s.onnx source=frames/ fps=25 \
          json=true quiet=true
---

## Synopsis

```bash
libreyolo track source=<video|directory|image|a.jpg,b.jpg> [model=<name|path>] [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `tracker=botsort`
and `--tracker botsort` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `source` | | Video file, image directory, image path, or a comma-separated `a.jpg,b.jpg` image list. Required |
| `model` | `yolox-s` | Model name or path; an exported artifact works too |
| `tracker` | `bytetrack` | `bytetrack`, `botsort`, `ocsort` or `deepocsort`. The `.yaml` spellings select the same built-in trackers |
| `track_conf` | `0.25` | Confidence threshold of the tracker's first association stage |
| `conf` | | Detection threshold. Unset, the detector runs at the tracker's own low threshold so weak boxes stay available for recovery |
| `iou` | `0.45` | NMS IoU threshold |
| `imgsz` | | Input image size: `640` (square) or `480x640` (HxW). The model's own input size when unset |
| `classes` | | Filter by class IDs, e.g. `[0,2,5]`. A bare integer is accepted |
| `max_det` | `300` | Max detections per frame |
| `save` | `false` | Save the annotated video |
| `show` | `false` | Display tracked frames; `q` stops |
| `vid_stride` | `1` | Process every N-th frame |
| `fps` | `30` | Frame rate assumed for image sequences; ignored for videos |
| `output_path` | | Output video path. Default `runs/track/<name>.mp4`, numbered so a run never overwrites another |
| `color_format` | `auto` | Input color: `auto`, `rgb`, `bgr` |
| `device` | `auto` | Device: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | JSON output to stdout, one object per frame |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `false` | Verbose stderr output |
| `help_json` | `false` | Dump command schema as JSON and exit |

## Examples

<code-tabs name="examples" />

## Notes

Each invocation tracks one source from its first frame, so there is no
`persist` option: track IDs never carry over between commands. Live sources are
rejected with `config_unsupported`: webcams and streams. Use a video file, an image directory or an image list.

The human output is one line per frame, such as
`frame 12 video.mp4: 1920x1080 3 tracks (ids 1, 2, 5)`, and stderr ends with the
frame count and the number of unique track IDs. With `json=true`, each frame is
one JSON object carrying `tracker`, `frame_index` and a `detections` list in
which every box has `track_id`, `class`, `class_id`, `confidence` and
`bbox_xyxy`.

A model whose task has no boxes to associate, or a model type without
`track()`, exits with `config_unsupported`. Exit codes follow
[`libreyolo predict`](/docs/cli/predict): `0` on success, `2` for a usage or
configuration error, `3` when the source cannot be found, `4` when the model
cannot be loaded, and `1` for other runtime failures.

The trackers, their configs and the Python `track()` call are covered in
[object tracking](/docs/tasks/object-tracking).
