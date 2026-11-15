---
title: Optical flow
seo_title: "Optical flow estimation in LibreYOLO"
description: "Estimate dense optical flow between video frames in LibreYOLO with RAFT, SEA-RAFT, WAFT, FlowSeek, DPFlow, UniMatch, NeuFlow v2 or MEMFOF, save .flo files and validate end-point error."
lead: "Optical flow predicts, for every pixel of one frame, where it moves in the next frame. LibreYOLO exposes it as the flow task, new in 1.8.0: a pair of frames in, one dense field of pixel displacements out, with eight families behind the same call."
keywords: [optical flow python, raft optical flow, sea-raft, dense motion estimation, flo file, end-point error, video motion field]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Two frames
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)           # (H, W, 2) float32, (dx, dy) in pixels
        print(flow.mean_magnitude)       # mean displacement in pixels
    - label: A video or a frame list
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")

        # N frames give N - 1 results; result i is the flow from frame i to i + 1.
        for result in model("video.mp4", stream=True, vid_stride=2):
            print(result.flow.max_magnitude)
    - label: Views and raw fields
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")
        frames = ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]

        model(frames, save=True)                    # colour wheel (default)
        model(frames, save=True, arrows=True)       # arrows over the frame
        model(frames, save=True, warp=True)         # warp check
        results = model(frames, save=True, save_flow=True)   # also writes .flo

        results[0].flow.save("pair0.npy")           # .flo or .npy
  cli:
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSEARAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreSEARAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
        libreyolo val model=LibreSEARAFTm-flow.pt data=my_flow.yaml
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # mean end-point error, pixels
        print(metrics["metrics/px1"])      # share of pixels under 1 px error
        print(metrics["metrics/px3"])
        print(metrics["metrics/fl_all"])   # KITTI outlier rate
---

## Definition

The `flow` task takes a sequence of frames and predicts, for each consecutive
pair, the displacement of every pixel from frame `t` to frame `t + 1`. Its
aliases `optical-flow`, `optical_flow` and `opticalflow` resolve to it, and
checkpoints carry the `-flow` suffix.

A prediction fills `result.flow`, a `FlowField` holding an `(H, W, 2)` float32
array on the original canvas of the first frame. `flow.data[y, x]` is
`(dx, dy)` in pixels, `+x` to the right and `+y` downwards. `.dx`, `.dy`,
`.magnitude`, `.angle`, `.max_magnitude` and `.mean_magnitude` read it, and
`.warp(image)` warps an image back with the field. `flow.confidence` is an
optional `(H, W)` map in `[0, 1]`, filled only by families that predict one
(SEA-RAFT, from its uncertainty head); the others leave it `None`.
`flow.next_frame` holds the second frame for the warp view. `result.boxes` is
`None`.

## Models

Eight families serve `flow`. None has a model page yet; load them by checkpoint
name, as in the snippets below.

| Family | Sizes | Weights licence |
|---|---|---|
| SEA-RAFT | s, m, l | BSD-3-Clause |
| RAFT | s, l, plus Sintel and KITTI variants | BSD-3-Clause |
| NeuFlow v2 | s, plus Sintel and Things variants | Apache-2.0 |
| WAFT | t, s, m, l | BSD-3-Clause; `t` uses DINOv3 and carries the DINOv3 License |
| FlowSeek | t, s, m, l | Apache-2.0 for t and s; m and l are non-commercial (CC BY-NC, Depth Anything V2 encoder) |
| DPFlow | m, plus Sintel, KITTI and Spring variants | Non-commercial (CC BY-NC-SA 4.0) |
| UniMatch | s, l, plus Sintel and KITTI variants | MIT |
| MEMFOF | m, plus five variants | BSD-3-Clause |

Checkpoints follow `Libre<Family><size>-flow.pt`, for example
`LibreSEARAFTm-flow.pt`, `LibreRAFTl-flow-kitti.pt` or
`LibreNeuFlowv2s-flow.pt`. The non-commercial weights print a notice when they
load. NeuFlow v2 is the small edge model. MEMFOF reads three frames per pair,
and the runner repeats the edge frames at the start and end of a sequence. All
eight have recorded parity against their upstream code, and their weights are
mirrored on the LibreYOLO Hugging Face organization.

## Predict

<code-tabs name="predict" />

A single image raises: flow needs at least two frames. A list or tuple of
images, a directory sorted by name, a video or a stream is one sequence, and
`N` frames give `N - 1` results. Result `i` carries the flow from frame `i` to
frame `i + 1`, with frame `i` as its `orig_img`. On videos, `vid_stride` sets
the frame gap, and `stream=True` yields one result per pair.

There are three views. The default is the Middlebury colour wheel: hue is
direction, saturation is magnitude, and zero motion is white. `arrows=True`
draws the frame with one arrow every 16 pixels. `warp=True` shows frame `t`
next to frame `t + 1` warped back onto it, so a correct field makes both panels
match. `save=True` writes the chosen view, and `save_flow=True` also writes the
raw field of every pair as a Middlebury `.flo` file named
`<frame stem>_<frame index>.flo`. `FlowField.save(path)` writes `.flo` or `.npy`
directly.

From the shell, pass the frames as a comma-separated `source`, and add
`arrows=true`, `warp=true` or `save_flow=true`:

<code-tabs name="cli" />

See [prediction](/docs/predict) for sources and streaming, and
[`libreyolo predict`](/docs/cli/predict) for every CLI flag.

## Dataset format

A flow dataset is a set of frame sequences. Frames live under
`images/<split>/<sequence>/` and are ordered by file name; each consecutive
pair inside one sequence directory is a sample, whose ground truth sits under
`flows/` with the same relative path and the first frame's stem.

```text
dataset/
  data.yaml
  images/
    val/alley_1/frame_0001.png
    val/alley_1/frame_0002.png
    val/alley_1/frame_0003.png
  flows/
    val/alley_1/frame_0001.flo
    val/alley_1/frame_0002.flo
```

```yaml
path: dataset
val: images/val
flows_dir: flows
flow_format: flo
```

`flow_format` is `flo` (Middlebury, the default), `kitti` (16-bit RGB PNG with
the blue channel as the validity mask) or `npy` (an `(H, W, 2)` float array in
which non-finite values are invalid). An optional per-pair mask
`<stem>_valid.png` marks valid pixels as nonzero; `flow_valid_suffix` changes
the suffix. The last frame of a sequence has no flow file, and frames in
different directories never pair.

## Train

Flow ships inference only. `train()` on a flow family raises
`NotImplementedError`.

## Validate

`val()` compares predicted and ground-truth vectors directly on the valid
pixels, with no alignment.

<code-tabs name="val" />

`metrics/epe` is the mean end-point error, the Euclidean distance between the
predicted and true vectors in pixels, and lower is better. `metrics/px1` and
`metrics/px3` are the fraction of valid pixels whose error is under 1 and 3
pixels. `metrics/fl_all` is the KITTI outlier rate: pixels whose error exceeds
both 3 pixels and 5 percent of the true magnitude. Best-checkpoint fitness is
`metrics/px1`, because it increases with quality.

## Export

Flow export is blocked for every format until the two-input graph and the
backend rescale contract are implemented for a family. See the
[export matrix](/docs/reference/export-matrix).
