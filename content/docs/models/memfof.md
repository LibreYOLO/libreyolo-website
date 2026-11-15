---
title: MEMFOF
families: [memfof]
seo_title: "MEMFOF optical flow in Python: memory-efficient multi-frame flow"
description: "Run MEMFOF in LibreYOLO for memory-efficient three-frame optical flow on Full HD video. One network with six checkpoints, BSD-3-Clause."
lead: "MEMFOF is a memory-efficient multi-frame optical flow network from the MSU Graphics & Media Lab (ICCV 2025 Highlight). It reads three frames, t-1, t and t+1, and predicts the flow of the middle frame, aimed at Full HD video. It is the one flow family in LibreYOLO that reads a three-frame window."
keywords: [MEMFOF, multi-frame optical flow, optical flow, Full HD optical flow, video optical flow, dense optical flow, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMEMFOFm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreMEMFOFm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMEMFOFm-flow.pt data=my_flow.yaml
---

MEMFOF is a RAFT descendant that predicts the backward (`t` to
`t-1`) and forward (`t` to `t+1`) flow of the middle frame in one pass.
LibreYOLO reports the forward flow as `Results.flow`; the backward flow is
computed but kept out of the task contract for now. The runner hands it the
`(t-1, t, t+1)` window and repeats frame 0 as `t-1` at the start of a
sequence, so you still pass an ordinary list, folder or video. Pick it for
real-world video at high resolution.

## Install

MEMFOF needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreMEMFOF` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. MEMFOF fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

One architecture (ResNet-34 FPN), size `m`, with six checkpoints:

| File | Upstream checkpoint |
| --- | --- |
| `LibreMEMFOFm-flow.pt` | MEMFOF-Tartan-T-TSKH, recommended by the authors for real-world video (canonical) |
| `LibreMEMFOFm-flow-tartan-t.pt` | an earlier published training stage |
| `LibreMEMFOFm-flow-tartan.pt` | an earlier published training stage |
| `LibreMEMFOFm-flow-kitti.pt` | KITTI fine-tune |
| `LibreMEMFOFm-flow-sintel.pt` | Sintel fine-tune |
| `LibreMEMFOFm-flow-spring.pt` | Spring fine-tune |

Frames run at native resolution with upstream's replicate padding to a
multiple of 16. Frames whose long side exceeds `imgsz` (default 1920, MEMFOF's
Full HD target) are downscaled first and the field is scaled back. Inference is
fp32 with 8 refinement iterations, upstream's defaults.

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

MEMFOF is the work of the
[MSU Graphics & Media Lab](https://github.com/msu-video-group/memfof). The
upstream inference code is vendored under its BSD-3-Clause licence; its
attention modules come from GMA under the WTFPL. The upstream checkpoints on
Hugging Face are tagged bsd-3-clause but ship no licence file, so LibreYOLO
mirrors the six converted checkpoints under the code repository's BSD-3-Clause
licence, without gating. BSD-3-Clause allows commercial and closed-source use
as long as the copyright notice, conditions and disclaimer are kept.

</provenance-box>

## Citation

<citation-block />
