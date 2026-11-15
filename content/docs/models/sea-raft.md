---
title: SEA-RAFT
families: [searaft]
seo_title: "SEA-RAFT optical flow in Python: fast RAFT with a confidence map, BSD-3"
description: "Run SEA-RAFT in LibreYOLO for dense optical flow between frames, with a per-pixel confidence map from its uncertainty head. Sizes s, m and l, BSD-3-Clause."
lead: "SEA-RAFT (Simple, Efficient, Accurate RAFT) is a RAFT descendant by Wang, Lipson and Deng (ECCV 2024). It predicts dense optical flow between two frames and, unlike the other flow families in LibreYOLO, also a per-pixel confidence map. LibreYOLO ports the upstream inference code and mirrors its S and M checkpoints."
keywords: [SEA-RAFT, SEARAFT, optical flow, dense optical flow, RAFT, flow confidence, optical flow uncertainty, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSEARAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreSEARAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSEARAFTm-flow.pt data=my_flow.yaml
---

SEA-RAFT keeps RAFT's all-pairs correlation pyramid, regresses an
initial flow directly, refines it with a ConvNeXt update block and is trained
with a mixture-of-Laplace loss. The predicted mixture variance is what
LibreYOLO turns into `flow.confidence`. It is the family the
[optical flow task page](/docs/tasks/optical-flow) uses in its examples; pick it
when you want a per-pixel confidence next to the flow.

## Install

SEA-RAFT needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreSEARAFT` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. `flow.confidence` is an `(H, W)` map in `[0, 1]` derived from the
predicted mixture variance: higher means the network is more certain of the
vector at that pixel. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

| Size | Encoders | Refinement iterations | File |
| --- | --- | ---: | --- |
| `s` | ResNet18 | 4 | `LibreSEARAFTs-flow.pt` |
| `m` | ResNet34 | 4 | `LibreSEARAFTm-flow.pt` |
| `l` | ResNet34, the `m` weights | 12 | `LibreSEARAFTl-flow.pt` |

Upstream ships no separate L weights: `l` is the `m` checkpoint run with 12
iterations instead of 4. All three are upstream's
Tartan-C-T-TSKH-spring540x960 checkpoints. `LibreSEARAFT` defaults to size
`m`. Frames up to 960 px on the long side, the training crop width, run at
native resolution padded to a multiple of 8; larger frames are downscaled and
the field is scaled back. `imgsz` changes that cap.

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

SEA-RAFT is the work of Wang, Lipson and Deng at the
[Princeton Vision & Learning Lab](https://github.com/princeton-vl/SEA-RAFT).
An inference subset of the upstream code is vendored under its BSD-3-Clause
licence. Upstream publishes the S and M weights without a separate weight
licence; LibreYOLO mirrors them under the repository's BSD-3-Clause licence,
without gating. BSD-3-Clause allows commercial and closed-source use as long
as the copyright notice, conditions and disclaimer are kept.

</provenance-box>

## Citation

<citation-block />
