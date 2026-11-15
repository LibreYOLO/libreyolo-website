---
title: DPFlow
families: [dpflow]
seo_title: "DPFlow optical flow in Python: dual-pyramid flow up to 8K"
description: "Run DPFlow in LibreYOLO for dense optical flow on high-resolution frames, with a pyramid that adapts to the input size. Apache-2.0 code; weights are non-commercial."
lead: "DPFlow is an adaptive optical flow estimator by Henrique Morimitsu and co-authors (CVPR 2025). Its dual-pyramid encoder adapts its depth to the input size, so a model trained at low resolution generalises to 4K and 8K frames. LibreYOLO ports it from ptlflow; the weights are licensed CC BY-NC-SA 4.0, for non-commercial use only."
keywords: [DPFlow, optical flow, high-resolution optical flow, 4K optical flow, dual pyramid, ptlflow, dense optical flow, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDPFlowm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreDPFlowm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDPFlowm-flow.pt data=my_flow.yaml
---

DPFlow chooses the number of pyramid levels from the image diagonal:
three levels around 1K, one more per doubling. LibreYOLO keeps that rule and
runs at the native frame resolution by default, which is where the adaptive
depth matters. Pick it for research on high-resolution footage; its weights
are non-commercial.

## Install

DPFlow needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreDPFlow` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. DPFlow fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

One architecture, size `m`, with four checkpoints:

| File | Training stage |
| --- | --- |
| `LibreDPFlowm-flow.pt` | FlyingThings3D (`things`), the generalisation checkpoint (canonical) |
| `LibreDPFlowm-flow-sintel.pt` | Sintel fine-tune |
| `LibreDPFlowm-flow-kitti.pt` | KITTI fine-tune |
| `LibreDPFlowm-flow-spring.pt` | Spring fine-tune |

The iteration schedule is fixed at upstream's test setting, 4 refinement
iterations per pyramid level. Every pair is estimated independently: the
upstream warm start across frames is not used. `imgsz` caps the long side
(default 8192, in practice no cap); frames above it are downscaled and the
field is scaled back to the original canvas.

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

DPFlow is the work of Henrique Morimitsu and co-authors, released in
[ptlflow](https://github.com/hmorimitsu/ptlflow). The code and the weights
carry different licences.

- The code is **Apache-2.0**: ptlflow's `models/dpflow` is vendored with
  timm-derived helpers (Apache-2.0) and an input padder adapted from RAFT
  (BSD-3-Clause).
- The weights are **CC BY-NC-SA 4.0**: **non-commercial use only**,
  attribution required, and anything you build from them and share must carry
  the same licence. The DPFlow README adds that the weights are for academic
  and research purposes only, because of the training datasets' licences.
  Treat all four checkpoints as research weights.

LibreYOLO mirrors the weights with the licence text and a non-commercial
banner, without gating, and prints a notice when they load.

</provenance-box>

## Citation

<citation-block />
