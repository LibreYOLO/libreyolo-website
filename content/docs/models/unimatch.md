---
title: UniMatch
families: [unimatch]
seo_title: "UniMatch GMFlow optical flow in Python: global matching, MIT"
description: "Run UniMatch (GMFlow) in LibreYOLO for dense optical flow by global matching. Sizes s and l plus Sintel and KITTI fine-tunes, MIT licensed."
lead: "UniMatch unifies optical flow, stereo and depth estimation with one transformer matcher (TPAMI 2023); GMFlow is its flow model. LibreYOLO ships the optical flow task with the GMFlow checkpoints, under MIT."
keywords: [UniMatch, GMFlow, GMFlow+, optical flow, global matching, dense optical flow, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreUniMatchl-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreUniMatchl-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreUniMatchl-flow.pt data=my_flow.yaml
---

GMFlow treats optical flow as a global matching problem: a transformer
compares features of the two frames and a softmax over all positions gives the
match, optionally refined by local regression. The vendored upstream code also
contains the stereo and depth paths, but only flow is wired in LibreYOLO. Pick
`l` for real-world footage, as upstream recommends.

## Install

UniMatch needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreUniMatch` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. UniMatch fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

| Size | Upstream checkpoint | File |
| --- | --- | --- |
| `s` | `gmflow-scale1-things`: global matching at 1/8 (Chairs, then Things) | `LibreUniMatchs-flow.pt` |
| `l` | `gmflow-scale2-regrefine6-mixdata`: GMFlow+ at two scales with 6 refinement steps (canonical) | `LibreUniMatchl-flow.pt` |
| `l` | Sintel fine-tune | `LibreUniMatchl-flow-sintel.pt` |
| `l` | KITTI-2015 fine-tune | `LibreUniMatchl-flow-kitti.pt` |

Only `l` has Sintel and KITTI fine-tunes. `LibreUniMatch` defaults to size
`l`. Preprocessing reproduces upstream `inference_flow`: portrait frames are
transposed to landscape, and both frames are resized so each side rounds up to
the size's padding factor (16 for `s`, 32 for `l`). The field is resized back
and its vectors rescaled. One deliberate fix over upstream: for portrait frames
upstream leaves `dx` and `dy` exchanged after transposing back; LibreYOLO swaps
them so the field follows the flow contract.

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

UniMatch is the work of the
[Autonomous Vision Group](https://github.com/autonomousvision/unimatch)
(University of Tübingen). The upstream `unimatch` package is
vendored verbatim under its MIT licence. Upstream publishes the GMFlow
checkpoints without a separate weight licence, so LibreYOLO hosts them under
the releasing project's MIT licence: an implied grant, LibreYOLO's
interpretation and not confirmed by the publisher. No gating. MIT allows
commercial and closed-source use as long as the copyright and licence notice
travel with copies.

</provenance-box>

## Citation

<citation-block />
