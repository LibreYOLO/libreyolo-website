---
title: WAFT
families: [waft]
seo_title: "WAFT optical flow in Python: warping-alone field transforms, BSD-3"
description: "Run WAFT in LibreYOLO for dense optical flow without a cost volume. Sizes t, s, m and l by encoder; s, m and l BSD-3-Clause, t under the DINOv3 License."
lead: "WAFT, Warping-Alone Field Transforms, is an optical flow method by Wang and Deng (ICLR 2026). It replaces RAFT's cost volume with high-resolution feature warping and a ViT refinement loop. LibreYOLO ports the upstream code and mirrors its four zero-shot checkpoints, one per feature encoder."
keywords: [WAFT, Warping-Alone Field Transforms, optical flow, dense optical flow, DINOv3, Depth Anything V2, Twins-SVT, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreWAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreWAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreWAFTm-flow.pt data=my_flow.yaml
---

WAFT drops the correlation volume that RAFT and its descendants
build, and instead warps high-resolution features of the second frame with the
current flow estimate before each refinement step. All four checkpoints are
upstream's zero-shot models (TartanAir, then Chairs, then Things). Pick `m`,
the a1 checkpoint the upstream README recommends for downstream use, unless
you have a reason to choose a specific encoder.

## Install

WAFT needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreWAFT` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. WAFT fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

| Size | Upstream checkpoint | Parameters | File |
| --- | --- | ---: | --- |
| `t` | a2, DINOv3 ViT-S/16 encoder | 56.5 M | `LibreWAFTt-flow-dinov3.pt` |
| `s` | a2, Depth Anything V2 ViT-S | 56.9 M | `LibreWAFTs-flow-dav2.pt` |
| `m` | a1, Depth Anything V2 ViT-S | 64.2 M | `LibreWAFTm-flow.pt` |
| `l` | a2, Twins-SVT-Large encoder | 136.0 M | `LibreWAFTl-flow-twins.pt` |

The encoder is the weight variant in the file name, and each size exists only
under the name in the table; any other combination raises with the published
name. `LibreWAFT` defaults to size `m`. Prediction runs at native resolution:
the network pads the pair to a multiple of 112, 32 or 16 depending on the
encoder and crops the flow back. Five refinement iterations are baked in, the
upstream evaluation default. `imgsz` caps the long side (default 8192, in
practice no cap).

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

WAFT is the work of Wang and Deng at the
[Princeton Vision & Learning Lab](https://github.com/princeton-vl/WAFT). The
upstream code is vendored under its BSD-3-Clause licence. Upstream publishes no
separate weight licence, so LibreYOLO mirrors the checkpoints under the
repository's BSD-3-Clause licence plus the licence of the encoder each one
embeds, without gating.

- `s`, `m` and `l` are BSD-3-Clause. Their encoders are Apache-2.0 (Depth
  Anything V2 Small for `s` and `m`, a Twins-SVT-Large copy derived from timm
  for `l`). Commercial and closed-source use is allowed.
- `t` contains a DINOv3 ViT-S/16 fine-tuned by the WAFT authors, so Meta's
  **DINOv3 License** applies to it: a custom licence, not OSI-approved, with
  its own use restrictions and trade-control terms. Read it before using `t`
  commercially.

</provenance-box>

## Citation

<citation-block />
