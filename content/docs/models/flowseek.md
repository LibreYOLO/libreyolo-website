---
title: FlowSeek
families: [flowseek]
seo_title: "FlowSeek optical flow in Python: depth foundation model priors"
description: "Run FlowSeek in LibreYOLO for dense optical flow with a Depth Anything V2 encoder and motion bases. Sizes t and s are Apache-2.0; m and l are non-commercial."
lead: "FlowSeek is an optical flow network by Poggi and Tosi (ICCV 2025) that adds a frozen Depth Anything V2 encoder and depth-derived motion bases to a SEA-RAFT style recurrent network. LibreYOLO ports it with four sizes; t and s are Apache-2.0, m and l are for non-commercial use only."
keywords: [FlowSeek, optical flow, dense optical flow, Depth Anything V2, motion bases, SEA-RAFT, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFlowSeekt-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreFlowSeekt-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFlowSeekt-flow.pt data=my_flow.yaml
---

FlowSeek borrows geometry knowledge from a depth foundation model:
the Depth Anything V2 encoder features and a set of motion bases computed from
depth feed the recurrent flow update. The hosted checkpoints are upstream's
final generalist fine-tune (TartanAir, Chairs, Things, then a mixed
Sintel, KITTI, HD1K and Things stage), the one upstream evaluates on Spring and
LayeredFlow. Pick `t` or `s` for commercial work; `m` and `l` carry a
non-commercial encoder.

## Install

FlowSeek needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreFlowSeek` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. FlowSeek fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

Upstream trains two networks; `s` and `l` reuse the `t` and `m`
weights with more refinement iterations, as upstream's evaluation configs do.

| Size | Context net | Depth Anything V2 encoder | Iterations | Weights licence |
| --- | --- | --- | ---: | --- |
| `t` | ResNet-18 | ViT-S | 4 | Apache-2.0 |
| `s` | ResNet-18 | ViT-S | 12 | Apache-2.0 |
| `m` | ResNet-34 | ViT-B | 4 | CC BY-NC 4.0, non-commercial |
| `l` | ResNet-34 | ViT-B | 12 | CC BY-NC 4.0, non-commercial |

Files are `LibreFlowSeek<size>-flow.pt`. `LibreFlowSeek` defaults to size `m`,
so pass a `t` or `s` checkpoint explicitly if you need commercial-use weights.
Inference runs at native resolution, padded to a multiple of 8 inside the
network, and each side of the padded canvas must be at least 128 px. `imgsz`
caps the long side.

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

FlowSeek is the work of Poggi and Tosi at the
[University of Bologna](https://github.com/mattpoggi/flowseek). Its code is
Apache-2.0 with BSD-3-Clause parts from SEA-RAFT, vendored in LibreYOLO.

The checkpoint licence depends on the size, because each checkpoint embeds a
Depth Anything V2 encoder:

- `t` and `s` embed the ViT-S encoder and are **Apache-2.0**: commercial and
  closed-source use is allowed.
- `m` and `l` embed the ViT-B encoder, which is **CC BY-NC 4.0**: for
  **non-commercial use only**, with attribution. Their model cards carry a
  non-commercial banner and LibreYOLO prints a notice when they load.

Upstream publishes no separate checkpoint licence, so the repository's
Apache-2.0 licence is applied as an implied grant, limited by the encoder
licence. The weights are mirrored on the LibreYOLO Hugging Face organization
without gating.

</provenance-box>

## Citation

<citation-block />
