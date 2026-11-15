---
title: RAFT
families: [raft]
seo_title: "RAFT optical flow in Python: dense motion between frames, BSD-3"
description: "Run RAFT in LibreYOLO for dense optical flow between video frames. Small and large checkpoints plus Sintel and KITTI fine-tunes, BSD-3-Clause."
lead: "RAFT is the recurrent all-pairs optical flow network by Teed and Deng (ECCV 2020). It takes two frames and returns, for every pixel of the first, its displacement to the second. LibreYOLO ports the upstream inference code and mirrors the four published checkpoints for the flow task."
keywords: [RAFT, RAFT optical flow, optical flow, dense optical flow, Recurrent All-Pairs Field Transforms, raft-things, raft-sintel, raft-kitti, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRAFTl-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreRAFTl-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRAFTl-flow.pt data=my_flow.yaml
---

RAFT builds a 4D correlation volume between all pixel pairs of the two
frames and refines one flow field with a recurrent update block. It is the
reference most later flow networks start from, SEA-RAFT, FlowSeek and MEMFOF
among them. Pick it when you want the well-known baseline, or the KITTI or
Sintel fine-tune for footage close to those benchmarks.

## Install

RAFT needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreRAFT` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. RAFT fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

| Size | Upstream checkpoint | File |
| --- | --- | --- |
| `s` | raft-small | `LibreRAFTs-flow.pt` |
| `l` | raft-things (canonical) | `LibreRAFTl-flow.pt` |
| `l` | raft-sintel fine-tune | `LibreRAFTl-flow-sintel.pt` |
| `l` | raft-kitti fine-tune | `LibreRAFTl-flow-kitti.pt` |

`LibreRAFT` defaults to size `l`. Inference runs at the native frame
resolution, padded to a multiple of 8 as upstream's `InputPadder` does, with a
fixed 20 refinement iterations (upstream `demo.py`). `imgsz` caps the long
side: larger frames are downscaled and the field is scaled back to the
original canvas.

Training is not offered: `train()` raises `NotImplementedError`.

## Validate

`val()` runs the shared flow validator on a flow dataset (frames per sequence
folder, Middlebury `.flo`, KITTI 16-bit PNG or `.npy` ground truth; see the
[dataset format](/docs/tasks/optical-flow#dataset-format)). It reports the mean
end-point error `metrics/epe`, `metrics/px1` and `metrics/px3` (share of valid
pixels under 1 and 3 px of error) and the KITTI outlier rate
`metrics/fl_all`.

<code-tabs name="val" />

## Export

<export-matrix />

Export is not available yet for any format. Optical flow export is blocked
until the two-input graph (`image1`, `image2`) and the backend vector-rescale
contract of the flow task are implemented for a family.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

RAFT is the work of Teed and Deng at the
[Princeton Vision & Learning Lab](https://github.com/princeton-vl/RAFT). The
upstream inference code is vendored under its BSD-3-Clause licence, and the
four checkpoints from the upstream download script are mirrored on the
LibreYOLO Hugging Face organization under the same licence, without gating.
BSD-3-Clause allows commercial and closed-source use as long as the copyright
notice, conditions and disclaimer are kept.

</provenance-box>

## Citation

<citation-block />
