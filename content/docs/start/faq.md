---
title: FAQ
seo_title: "LibreYOLO FAQ"
description: "Short answers to the questions that cut across every LibreYOLO model: hardware, licensing, weights, devices, training, export coverage and the CLI."
lead: "Answers to questions that are not specific to one model family. Anything family-specific lives on that family's page."
keywords: [libreyolo faq, libreyolo gpu required, libreyolo license, libreyolo weights location, libreyolo cli, libreyolo offline]
last_verified: "1.5.0"
---

## Which model should I start with?

YOLOv9 for a CNN detector and RF-DETR for a transformer one. Both sit in the
flagship tier, which means features are designed and GPU-validated against them
before anything else. See [YOLOv9](/docs/models/yolov9) and
[RF-DETR](/docs/models/rf-detr), or [all models](/docs/models) for the rest.

## Do I need a GPU?

No. Every model runs on CPU, and everything in the
[quickstart](/docs/quickstart) is written to run there. A GPU changes how long
training and video inference take, not whether they work.

## How does LibreYOLO choose a device?

The default is `device="auto"`, which uses CUDA when PyTorch reports it
available, then Metal Performance Shaders when those are available, and CPU
otherwise. To pin it, pass `device` to the model or to `predict`, `train`, `val`
and `export`. It accepts `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, a bare integer
such as `0`, or a digit string; the last two expand to `cuda:<n>`.

`libreyolo checks` prints the Torch build, its CUDA and cuDNN versions, and
every GPU it can see. If that command shows no CUDA, the PyTorch wheel is a CPU
build; [install](/docs/install) covers replacing it.

## Where do downloaded weights go?

Into `weights/` relative to the working directory. A model reference with no
directory component resolves there and is downloaded on first use; a reference
that includes a directory is used exactly as written and is never fetched. See
[checkpoints and weights](/docs/weights).

## Can I run with no network access?

Yes. Fetch the checkpoints once on a connected machine, copy the `weights/`
directory across, and nothing will reach the network again. A shared read-only
path also works, since a reference containing a directory is taken literally.
Datasets resolve under `~/datasets`, or under `LIBREYOLO_DATASETS_DIR`.

## Can I use LibreYOLO commercially?

The code is MIT licensed. Pretrained weights are a separate question: they can
inherit terms from the project or dataset they came from, and those terms are
not uniform even within one family. The license on the specific Hugging Face
repository is authoritative, and every model page carries a licensing section
that reproduces it. Where weights are restricted, LibreYOLO prints the
restriction before the download starts.

## Can I load a checkpoint from another project?

Usually, by passing its path to `LibreYOLO()`. Recognized upstream layouts are
converted at load time, keeping their class count and names, and a LibreYOLO
checkpoint is written next to the source. [Import existing
weights](/docs/migrate) covers what is recognized and what needs a conversion
script.

## Why does train raise NotImplementedError?

Because that family ships inference only, and the exception names the reason.
Predict, validate and, where supported, export all work; there is no training
loop for that architecture in LibreYOLO. The support tier in a model page's
header tells you before you try. See [core concepts](/docs/concepts).

## What does val return?

A plain dictionary, not an object. Detection keys include
`metrics/precision`, `metrics/recall`, `metrics/mAP50` and
`metrics/mAP50-95`. Other tasks return the keys that make sense for them, such
as `metrics/accuracy_top1` for classification or `metrics/PQ`, `metrics/SQ` and
`metrics/RQ` for panoptic segmentation.

## How do I run on a folder, a video or a webcam?

Pass it as the source. A file path is one image, a directory is every image in
it, a video path is a video, an integer is a webcam index, and an RTSP, RTMP,
TCP, UDP or HLS URL is a live stream. A `.streams` file lists several sources at
once. Live sources require `stream=True`, which yields one `Results` per frame
instead of building a list; the same flag is worth using for long videos and
large directories. Only YouTube page URLs need an extra, `libreyolo[stream]`.

## How do I keep only some classes?

Pass `classes` to `predict` with the class indices you want, for example
`classes=[0, 2]`. `conf` sets the confidence threshold, default `0.25`, and
`max_det` caps detections per image, default `300`.

## Does the CLI use flags or key=value pairs?

Key and value joined by an equals sign, for every command:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` accepts a path or a short name of the form `family-size`, optionally
with a task suffix, and `libreyolo models` lists every valid one. Diagnostic and
inventory commands also take `--json`, which prints the same data as a
machine-readable object on stdout.

## Can every model export to every format?

No. Coverage is per family and per task, not uniform, and each format has its
own extra to install. Each model page carries its family's export matrix; the
[export section](/docs/export) covers the formats themselves.

## What is the difference between segment, semantic and panoptic?

Three separate tasks. `segment` produces one mask per detected object.
`semantic` labels every pixel with a class and separates nothing into
instances. `panoptic` gives every pixel exactly one label, merging countable
things with amorphous stuff. They have different ground truth, different result
fields and different metrics, and a family supports whichever of them appears in
its task list.

## How do I train on my own classes?

Write a dataset YAML with `train`, `val` and `names`. Labels sit beside the
images in a parallel `labels/` tree, one `.txt` per image, with normalized
coordinates. `nc` is optional and must agree with `names` when present. Run
`libreyolo doctor <data.yaml>` first: it checks the dataset for problems and
exits non-zero when it finds errors, which makes it usable as a CI gate.

## Why does loading print a metadata warning?

Because the checkpoint does not carry complete v1.0 metadata. Loading continues
through a compatibility path, and the warning names exactly which keys are
missing. Run `libreyolo metadata path=<file>` to see what is there, and see
[checkpoints and weights](/docs/weights) for what the schema requires.

## An import stopped working after an upgrade. What changed?

Two class names were renamed for consistency: `LibreYOLORTDETR` became
`LibreRTDETR` and `LibreYOLORFDETR` became `LibreRFDETR`. The old names still
resolve and emit a `DeprecationWarning` pointing at the new one, so existing
code keeps running while you update it.
