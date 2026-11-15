---
title: Video Depth Anything
families: [video_depth_anything]
seo_title: "Video Depth Anything in Python: consistent depth for long videos"
description: "Use Video Depth Anything in LibreYOLO for temporally consistent relative depth across the frames of a video. Sizes s, b and l; s is Apache-2.0, b and l are non-commercial."
lead: "Video Depth Anything, by ByteDance, puts a temporal head on the Depth Anything V2 encoder so depth stays consistent from frame to frame. LibreYOLO runs a list of frames, a folder, a video or a stream as one sequence and returns one depth map per frame, all on one relative scale."
keywords: [Video Depth Anything, video depth estimation, temporally consistent depth, Depth Anything V2, relative depth, DINOv2]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVideoDepthAnythings-depth.pt")

        # One result per frame, streamed with at most one window of lag.
        for result in model("clip.mp4", stream=True):
            depth = result.depth_map.data      # (H, W) relative inverse depth
    - label: One image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVideoDepthAnythings-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)   # a one-frame sequence
        print(result.depth_map.data.shape)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreVideoDepthAnythings-depth.pt source=clip.mp4 save=True
---

## Install

Video Depth Anything needs no optional extra.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face mirrors on first use.

<code-tabs name="predict" />

Any multi-image source is one sequence: a list of frames, a directory sorted
by name, a video file or a stream. `N` frames give `N` results, each with a
`result.depth_map` on that frame's canvas. Inside, frames run in upstream's
overlapping 32-frame windows, and each window is scale and shift aligned to
the ones before it, so the whole sequence shares one relative scale. A single
image is a one-frame sequence; upstream pads it to a full window with copies
of itself.

The output follows LibreYOLO's shared depth contract: relative inverse depth,
where higher values are closer and the values have no metric unit. The input
size is upstream's default of 518 for every encoder; frames wider than 16:9
run proportionally smaller, as upstream does. See the
[depth estimation](/docs/tasks/depth-estimation) task page for the payload.

## Variants

Three encoder sizes, `s`, `b` and `l`, one checkpoint each. The licence
differs by size (see below), so `s` is the one to use in a commercial
product. Upstream's metric-depth checkpoints are not hosted, and the
converter refuses them.

Training is not offered: `train()` raises. Convert an upstream relative-depth
checkpoint with `weights/convert_video_depth_anything_weights.py` in the
library.

## Export

<export-matrix />

Export is not supported. The network reads a 32-frame window and the windows
are stitched on the host, which the single-image depth export contract does
not describe, so `export()` raises.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box>

Video Depth Anything is by ByteDance
([DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)).
The vendored code is Apache-2.0. The licence of the weights depends on the
size:

- `s` (Video-Depth-Anything-Small): Apache-2.0, commercial use allowed.
- `b` and `l` (Base and Large): CC BY-NC 4.0, **non-commercial only**.
  They ship with a non-commercial notice.

LibreYOLO mirrors the converted checkpoints, tensors unchanged, as
`LibreYOLO/LibreVideoDepthAnything{s,b,l}-depth`, not gated.

</provenance-box>

## Citation

<citation-block />
