---
title: NeuFlow v2
families: [neuflowv2]
seo_title: "NeuFlow v2 optical flow in Python: a 9M-parameter edge model, Apache-2.0"
description: "Run NeuFlow v2 in LibreYOLO for fast dense optical flow on edge devices. One 9M-parameter network with mixed, Sintel and Things checkpoints, Apache-2.0."
lead: "NeuFlow v2 is a high-efficiency optical flow network from the Northeastern University Field Robotics Lab, built for edge devices. It has about 9M parameters and runs on a fixed canvas; it is the edge model among LibreYOLO's flow families."
keywords: [NeuFlow v2, NeuFlow, optical flow, edge optical flow, real-time optical flow, dense optical flow, flo file]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")   # weights download on first use
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)    # mean displacement in pixels
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreNeuFlowv2s-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreNeuFlowv2s-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNeuFlowv2s-flow.pt data=my_flow.yaml
---

NeuFlow v2 pairs a shallow CNN encoder with two cross-attention layers
and a global softmax match at 1/16 scale, then a light convolutional refiner
at 1/16 and 1/8 scale and a learned 8x convex upsampler. Pick it when speed and
memory matter more than the last bit of accuracy, for example on video at
frame rate or on small hardware.

## Install

NeuFlow v2 needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face organization on first use.
The class `LibreNeuFlowv2` is also exported from `libreyolo` if you prefer to build the
model directly.

<code-tabs name="predict" />

Flow needs at least two frames: a single image raises. A list of images, a
directory sorted by name, a video or a stream is one sequence, and `N` frames
give `N - 1` results. `result.flow` is a `FlowField` with an `(H, W, 2)` float32
array on the original canvas of the first frame; `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` right and `+y` down. NeuFlow v2 fills no confidence map, so `flow.confidence` is `None`. `result.boxes` is
`None`. `save=True` writes the Middlebury colour wheel, `arrows=True` or
`warp=True` pick the other views, and `save_flow=True` also writes each pair's
raw field as a `.flo` file. The [optical flow task](/docs/tasks/optical-flow)
page describes the payload and views in full.

## Variants

One architecture, size `s`, with three training-data variants of the
same network:

| File | Upstream checkpoint |
| --- | --- |
| `LibreNeuFlowv2s-flow.pt` | `neuflow_mixed.pth`, trained on several datasets (canonical) |
| `LibreNeuFlowv2s-flow-sintel.pt` | `neuflow_sintel.pth` |
| `LibreNeuFlowv2s-flow-things.pt` | `neuflow_things.pth` |

Unlike the other flow families, NeuFlow v2 does not run at native resolution.
Both frames are stretch-resized to a fixed canvas, 432x768 (height x width) by
default as in upstream `infer.py`; `imgsz` accepts an int or a
`(height, width)` pair, each side a multiple of 16. The network runs one
refinement step at 1/16 and eight at 1/8, and the field is resized back to the
original canvas with each vector component multiplied by its axis resize
ratio.

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

NeuFlow v2 is the work of the
[Northeastern University Field Robotics Lab](https://github.com/neufieldrobotics/NeuFlow_v2).
The upstream model code is vendored under its Apache-2.0 licence. Upstream
commits its three checkpoints to the same Apache-2.0 repository with no
separate weight licence, so LibreYOLO mirrors them under Apache-2.0, without
gating. Apache-2.0 allows commercial and closed-source use; keep the licence
text and attribution notices on redistribution.

</provenance-box>

## Citation

<citation-block />
