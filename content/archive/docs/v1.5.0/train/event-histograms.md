---
title: Prophesee event histograms
seo_title: Prophesee event histograms with LibreYOLO
description: 'Development guide for LibreYOLO v1.6: convert Prophesee SDK histograms, prepare labels, train YOLO9 or RF-DETR, and run ONNX inference.'
lead: Use numerical positive and negative event-count planes as detector inputs, with Prophesee handling event acquisition and LibreYOLO handling training and inference.
keywords:
- Prophesee
- Metavision SDK
- OpenEB
- event histogram
- event camera object detection
- LibreYOLO v1.6
verification: Input workflow checked against the v1.6 development implementation. SDK API checked against Metavision 5.3.1; SDK runtime validation is pending.
snippets:
  install:
  - label: Install the development implementation
    language: bash
    code: |
      python -m pip install "libreyolo[rfdetr,onnx] @ git+https://github.com/LibreYOLO/libreyolo.git@event-histogram-input"
  pack:
  - label: Convert an existing numerical histogram
    language: python
    code: |
      import numpy as np


      def pack_histogram(frame, *, layout, polarity, count_divisor=1.0):
          """Convert known producer output to HWC positive/negative counts."""
          values = np.asarray(frame)
          if values.dtype.kind not in "uif" or values.ndim != 3:
              raise ValueError("Expected a numerical three-dimensional histogram")
          if layout == "CHW":
              values = values.transpose(1, 2, 0)
          elif layout != "HWC":
              raise ValueError("Declare layout as HWC or CHW")
          if values.shape[2] != 2 or min(values.shape[:2]) == 0:
              raise ValueError("Expected two nonempty polarity planes")
          if polarity == "negative_positive":
              values = values[..., [1, 0]]
          elif polarity != "positive_negative":
              raise ValueError("Declare the producer's polarity order")
          if not np.isfinite(count_divisor) or count_divisor <= 0:
              raise ValueError("count_divisor must be positive and finite")
          with np.errstate(over="ignore", invalid="ignore"):
              counts = values.astype(np.float32) * count_divisor
          if not np.isfinite(counts).all() or (counts < 0).any():
              raise ValueError("Counts must be finite, nonnegative float32 values")
          return np.ascontiguousarray(counts)


      # Replace this path with a numerical histogram from your producer.
      sdk_histogram = np.load("sdk-histogram.npy", allow_pickle=False)
      histogram = pack_histogram(
          sdk_histogram, layout="HWC", polarity="negative_positive",
          count_divisor=1.0,  # Counts already: no producer normalization to undo.
      )
      np.save("histogram.npy", histogram, allow_pickle=False)
  sdk:
  - label: Build one histogram with the SDK Core API
    language: python
    code: |
      import numpy as np
      from metavision_sdk_core import EventPreprocessor


      def histogram_from_sdk_chunks(chunks, *, width, height, start_us,
                                    window_us=40_000, scale=16.0):
          """chunks must cover one known window, using SDK EventCD arrays."""
          processor = EventPreprocessor.create_HistoProcessor(
              input_event_width=width,
              input_event_height=height,
              max_incr_per_pixel=1.0,  # One increment is one count.
              clip_value_after_normalization=scale,
              use_CHW=False,          # SDK output is HWC, negative then positive.
              scale_width=1.0,
              scale_height=1.0,
          )
          frame = processor.init_output_tensor()
          frame.fill(0)
          end_us = start_us + window_us
          for events in chunks:
              if len(events) == 0:
                  continue
              if ((events["t"] < start_us) | (events["t"] >= end_us)).any():
                  raise ValueError("Split events into the declared window first")
              if ((events["x"] < 0) | (events["x"] >= width)
                      | (events["y"] < 0) | (events["y"] >= height)).any():
                  raise ValueError("Event coordinates do not match the sensor canvas")
              if not np.isin(events["p"], [0, 1]).all():
                  raise ValueError("Expected SDK polarity values 0 and 1")
              processor.process_events(start_us, np.ascontiguousarray(events), frame)
          return np.ascontiguousarray(frame[..., [1, 0]], dtype=np.float32)
  dataset:
  - label: data.yaml
    language: yaml
    code: |
      path: /absolute/path/to/event-dataset
      train: images/train
      val: images/val
      names:
        0: person
      input_profile:
        format: event_histogram
        layout: HWC
        polarity: positive_negative
        encoding: counts
        scale: 16.0
        window_us: 40000
  labels:
  - label: labels/train/frame000001.txt
    language: text
    code: |
      0 0.5 0.5 0.25 0.6
  inspect:
  - label: Inspect the numerical input and save a visualization
    language: python
    code: |
      import numpy as np
      from PIL import Image
      from libreyolo.utils.event_histogram import load_histogram, visualize_histogram

      histogram = load_histogram("histogram.npy")
      print(histogram.shape, histogram.dtype)
      print(float(histogram.min()), float(histogram.max()))
      rgb = visualize_histogram(histogram, scale=16.0)
      Image.fromarray(rgb).save("histogram-view.png")
  yolo9:
  - label: Train YOLO9
    language: python
    code: |
      from libreyolo import LibreYOLO, LibreYOLO9

      model = LibreYOLO9(None, size="t", device="cpu")
      run = model.train(
          data="data.yaml", pretrained=False,
          epochs=100, batch=4, imgsz=128, workers=0, device="cpu",
      )
      trained = LibreYOLO(run["best_checkpoint"], device="cpu")
      trained.save("event-detector.pt")
  rfdetr:
  - label: Train RF-DETR from scratch
    language: bash
    code: |
      libreyolo train model=LibreRFDETRn.pt pretrained=False data=data.yaml epochs=100 batch=4 imgsz=128 workers=0 device=cpu project=runs name=event-rfdetr exist_ok=True
  - label: Save the selected RF-DETR checkpoint
    language: python
    code: |
      from libreyolo import LibreYOLO

      trained = LibreYOLO("runs/event-rfdetr/weights/best.pt", device="cpu")
      trained.save("event-detector.pt")
  predict:
  - label: Predict from a file or an array
    language: python
    code: |
      import numpy as np
      from libreyolo import LibreYOLO

      model = LibreYOLO("event-detector.pt", device="cpu")
      result = model.predict("histogram.npy", save=True)
      print(result.boxes.xyxy)
      print(result.boxes.conf)

      histogram = np.load("histogram.npy", allow_pickle=False)
      result = model.predict(histogram)
      results = model.predict([histogram, histogram.copy()])
  - label: Predict from the CLI
    language: bash
    code: |
      libreyolo predict model=event-detector.pt source=histogram.npy save=True
  val:
  - label: Evaluate the validation split
    language: python
    code: |
      from libreyolo import LibreYOLO

      model = LibreYOLO("event-detector.pt", device="cpu")
      metrics = model.val(data="data.yaml", split="val", batch=4, workers=0)
      print(metrics["metrics/mAP50-95"])
  onnx:
  - label: Export and load ONNX
    language: python
    code: |
      from libreyolo import LibreYOLO

      model = LibreYOLO("event-detector.pt", device="cpu")
      onnx_path = model.export(
          format="onnx", imgsz=128, dynamic=False, simplify=False,
          output_path="event-detector.onnx",
      )
      runtime = LibreYOLO(onnx_path, device="cpu")
      result = runtime.predict("histogram.npy", save=True)
---

**Availability: development documentation for `dev`, scheduled for release in
LibreYOLO v1.6.** The histogram implementation is currently in
[PR #865](https://github.com/LibreYOLO/libreyolo/pull/865), awaiting merge into
`dev`. Use the feature-branch install below until it merges. This functionality
is not part of the current PyPI release.

## Install the development version

Run this in the Python environment that will run the detector:

<code-tabs name="install" />

Once PR #865 is merged, replace `@event-histogram-input` with `@dev` in that
command. After v1.6 is released, install the corresponding PyPI version instead.

If your application already uses the Prophesee Metavision SDK, keep its event
acquisition and decoding setup. LibreYOLO does not need the SDK when it receives
prepared arrays or files. The SDK producer and detector can run in separate
environments and exchange `.npy` files. Passing an array directly requires both
packages in the same interpreter. For a new SDK installation, follow the publisher's
[OpenEB installation guide](https://docs.prophesee.ai/stable/installation/index.html).

## What crosses the SDK boundary

The workflow is:

```text
SDK event window → two numerical count planes → LibreYOLO detector → boxes
```

The supported input is a NumPy array with shape **`(height, width, 2)`**, or a
`.npy` file containing that array. Channel 0 holds positive counts; channel 1
holds negative counts. Values must be finite and nonnegative. Integer and
floating-point counts are accepted and processed as float32.

This is compatible with two-polarity numerical histograms after their layout,
channel order and scale are made explicit. It does not accept every image an
SDK can display. An RGB event visualization, a reconstructed intensity frame,
a signed difference image, a time surface or a multi-bin event cube does not
contain this same input representation. Regenerate two count planes when the
original information has been combined or discarded.

For example, a pixel with three positive events and one negative event is
stored as `[3, 1]`. A window with no events is a zero array with the complete
sensor dimensions.

## Use histograms you already have

Check the producer settings before converting. `HWC` means height, width,
channels; `CHW` means channels, height, width. Do not guess polarity order from
colors or from which channel has larger values.

Put this helper in your preprocessing script. It converts a declared producer
layout and polarity order, and reverses a known normalization divisor:

<code-tabs name="pack" />

For an SDK histogram calculated as `clip(counts / 16, 0, 1)`, use
`count_divisor=16.0` and `input_profile.scale: 16.0`. The resulting file contains
counts saturated at 16, which gives the same normalized detector input as raw
counts with that saturation level. If the SDK already stores counts, use
`count_divisor=1.0`.

Clipping cannot be undone. Counts clipped at 5 cannot reproduce a model input
that distinguishes counts up to 16. Regenerate the histogram or train with a
matching producer configuration. The example assumes sensor coordinates have
not already been rescaled; preserve that canvas for both inputs and labels.

## Create histograms with the Prophesee Core API

The SDK provides `EventPreprocessor.create_HistoProcessor` through its public
[Core Python API](https://docs.prophesee.ai/stable/api/python/core/bindings.html#metavision_sdk_core.EventPreprocessor.create_HistoProcessor).
The example below selects HWC output, uses one count per event, and reorders
the SDK's negative/positive planes into LibreYOLO's positive/negative order.
It retains counts up to the same `scale` used in the dataset YAML.

**The SDK example is checked against the published API, but has not been run
against an installed SDK in this validation environment.** The prepared-array
and file workflows in the following sections have been exercised with both
models.

<code-tabs name="sdk" />

Pass decoded SDK `EventCD` arrays with `x`, `y`, `p` and `t` fields. Preserve the
SDK dtype; an arbitrary Nx4 integer matrix is not an `EventCD` array. SDK
polarity 0 is negative and 1 is positive. `width` and `height` must describe the
coordinate system of those events.

`chunks` must contain all the events for one known window. Set `start_us` from
your acquisition or annotation timeline, and use the same duration for every
training and prediction sample. Do not infer either boundary from the first
or last observed event. The helper rejects chunks containing events outside
`[start_us, start_us + window_us)`.

The output tensor is cleared once per window and retained across its chunks.
For an empty window, pass an empty iterable and keep the resulting zero frame.
For a nonempty window, processing one combined batch and processing the same
events in several chunks should produce identical histograms. Check that
property with your SDK version before preparing the whole dataset.

Save the returned array with `np.save("histogram.npy", histogram,
allow_pickle=False)`, or pass it directly to a trained detector. RAW files and
live cameras remain the responsibility of the SDK application; this feature
does not add them as LibreYOLO prediction sources.

## Prepare the dataset and labels

Use one histogram and one YOLO detection label file per annotated event window:

```text
event-dataset/
  data.yaml
  images/
    train/frame000001.npy
    val/frame000002.npy
  labels/
    train/frame000001.txt
    val/frame000002.txt
```

Save the following as `data.yaml`, replacing `path` with your dataset location
and `names` with your classes. Here, 40,000 microseconds means a 40 ms event
window; 16 is the count saturation level. These are example settings, not a
recommended acquisition duration for every scene.

<code-tabs name="dataset" />

All six `input_profile` fields are required. LibreYOLO computes
`clip(counts / scale, 0, 1)` before resizing. It does not apply RGB/BGR conversion,
ImageNet normalization or division by 255. The scale must be positive and
finite in float32. The accumulation duration must be a positive integer.

Labels retain the standard `class_id cx cy width height` format, with all four
box values normalized to the **original histogram canvas**:

<code-tabs name="labels" />

The example labels one class-0 box centered in the image. Associate boxes with
the same event window as the histogram. Split by recording or scene to avoid
putting nearby windows into both training and validation. Missing or empty
label files mean no objects to the loader, so do not include unannotated windows
as negative samples unless absence of objects is known.

For other dataset conventions, see [Datasets](/docs/train/datasets).

## Inspect the input

<code-tabs name="inspect" />

The visualization uses red for positive activity, blue for negative activity
and black for zero activity. Inspect labels and predictions on this image, but
keep the numerical `.npy` array as the model input.

## Train a detector

Choose [YOLO9](/docs/models/yolov9) or [RF-DETR](/docs/models/rf-detr). Both examples
start with random weights and save the selected checkpoint as
`event-detector.pt`. They use a small 128-pixel canvas for an initial workflow
check; choose the final resolution and training duration for your data. Replace
`device="cpu"` with your training device when appropriate.

### YOLO9

<code-tabs name="yolo9" />

### RF-DETR

The CLI's `pretrained=False` route constructs the architecture without loading
RGB weights. The model filename selects its family and size.

<code-tabs name="rfdetr" />

YOLO9 preserves aspect ratio and pads with zeros. RF-DETR stretches to the
requested canvas and retains its patch/window divisibility requirements.
Histogram training supports horizontal and vertical flips; RGB color changes,
mosaic, mixup, random crops and affine/projective augmentation are disabled.
Unsupported explicit augmentation settings raise an error.

For transfer learning, load a compatible RGB checkpoint and call `train()` with
the histogram dataset, omitting `pretrained=False`. LibreYOLO adapts the input
convolution to two channels and preserves compatible weights. Those weights
still need event-data training. The checkpoint records whether its input layer
was randomly initialized or adapted from RGB.

Training currently requires one device and a fixed positive batch size.
Auto-batch, CUDA graph training, RF-DETR multi-scale training, LoRA,
distillation and quantized histogram models are outside this workflow.

## Reload, predict and validate

The saved model carries its input profile. Prediction therefore needs neither
the dataset YAML nor a repeated scale or polarity setting:

<code-tabs name="predict" />

Boxes are returned in original-canvas coordinates. A single input returns one
`Results`; a list returns a list. `save=True` draws detections on the separate
color visualization. Directories of `.npy` files are also supported.

Validate with the same dataset profile and a separate validation split:

<code-tabs name="val" />

A different scale, duration, encoding or channel order is a different input
contract. Validation and training reject a declared profile that does not
match the loaded checkpoint. Prediction cannot detect mislabeled numerical
arrays; keep the producer configuration consistent with the saved model.

## Export and run ONNX

<code-tabs name="onnx" />

The ONNX file preserves the input profile, so the LibreYOLO runtime applies the
same preprocessing. The graph itself consumes normalized float32 NCHW tensors
with two channels; it does not decode or accumulate events. In an installation
without PyTorch, use `libreyolo.backends.onnx.OnnxBackend` directly.

The initial histogram export path supports FP32 ONNX without embedded NMS.
Other export runtimes, TTA and tiling are not supported by this workflow.

## What has been verified

Both families have been exercised with prepared arrays and `.npy` files,
training runs, checkpoint reload, CLI prediction, saved visualizations and
native/ONNX comparisons. Prepared-data inference requires no SDK installation.
Small PEDRo examples were also converted and inspected, but these checks do
not establish full-dataset accuracy or a recommended training recipe.

OpenEB runtime execution, RAW decoding, live-camera operation and a larger
recording-separated evaluation remain unverified. The
[implementation PR](https://github.com/LibreYOLO/libreyolo/pull/865) records the
current validation scope. SDK integration details refer to the
[Metavision 5.3.1 preprocessing documentation](https://docs.prophesee.ai/stable/guides/events_preprocessing.html)
and the public OpenEB Core API at revision
[`9003b5416676e78ba994d912087486cfa94fae73`](https://github.com/prophesee-ai/openeb/tree/9003b5416676e78ba994d912087486cfa94fae73).
