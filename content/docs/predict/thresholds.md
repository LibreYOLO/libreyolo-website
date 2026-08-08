---
title: Thresholds and filtering
seo_title: "conf, iou and max_det in LibreYOLO"
description: "What conf, iou, max_det and classes actually do at predict time, which families ignore iou because they run no NMS, and why agnostic_nms is a no-op."
lead: "Four arguments decide which predictions survive: conf, iou, max_det and classes. Only two of them apply to every family, because a set predictor decodes a fixed query set and never runs NMS."
keywords:
  - yolo conf threshold
  - iou threshold nms
  - max_det
  - filter classes detection python
  - agnostic nms
  - nms free detr
  - detection confidence threshold
  - class filtering inference
last_verified: "1.5.0"
verification: "Defaults quoted from InferenceRunner.__call__ in libreyolo/models/base/inference.py. Per-family NMS behavior read from every module in libreyolo/postprocess/ and cross-checked against _is_nms_free_family in libreyolo/backends/base.py. Class filtering from InferenceRunner._apply_classes_filter and _wrap_results. agnostic_nms status from NOOP_PREDICT_KWARGS in libreyolo/utils/predict_args.py. Open-vocabulary handling from NMS_THRESHOLD in libreyolo/models/openvocab/base.py. Validation defaults from BaseModel.val."
snippets:
  basic:
    - label: The four arguments
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # keep predictions at or above this score
            iou=0.45,       # NMS overlap threshold, where NMS runs
            max_det=300,    # cap per image
            classes=None,   # or a list of class ids
        )
        print(len(result.boxes))
    - label: Sweeping conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Filtering to specific classes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Class ids index model.names. On COCO, 0 is person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Finding the id for a name
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou on a family that runs no NMS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # RF-DETR decodes a fixed query set, so iou changes nothing here.
        model = LibreYOLO("LibreRFDETRs.pt")

        loose = model(SAMPLE_IMAGE, iou=0.9)
        tight = model(SAMPLE_IMAGE, iou=0.1)

        # Same count either way. conf and max_det are the controls that work.
        print(len(loose.boxes), len(tight.boxes))
---

## The four arguments

| Argument | Default | Applies to |
|---|---|---|
| `conf` | `0.25` | Every family |
| `iou` | `0.45` | Families that run non-maximum suppression |
| `max_det` | `300` | Every family |
| `classes` | `None` | Every family |

<code-tabs name="basic" />

Two of these are universal and two are not, which is the single most useful
thing to know before tuning anything.

Validation uses different defaults on purpose: `val()` runs at `conf=0.001` and
`iou=0.6`, because average precision is computed over a full precision-recall
curve and a 0.25 cutoff would truncate it.

## conf

`conf` is the score below which a prediction is discarded. It applies to every
family, including the ones that never run NMS, and it is the first control to
reach for when there are too many or too few detections.

The default of `0.25` suits looking at pictures. Feeding a downstream system
usually wants it higher; measuring accuracy wants it far lower.

## iou

`iou` is the overlap above which non-maximum suppression removes the
lower-scoring of two boxes of the same class. It only means something if the
family runs suppression at all.

A set predictor decodes a fixed number of queries and takes the top scoring
ones. Duplicates are suppressed inside the architecture during training, not by
a postprocessing step, so there is no threshold to turn. These families accept
`iou` for API parity and ignore it:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR, and the end-to-end YOLOv9
head. Variants built on those decoders inherit the behavior.

<code-tabs name="nmsfree" />

Most of them say so in their postprocessing docstrings, but no warning is
raised at runtime, so a sweep over `iou` on RF-DETR produces a flat line rather
than an error. Faster R-CNN and Mask R-CNN are a slightly different case: both
already ran NMS inside the model, at a fixed upstream threshold that `iou` has
no supported way to change.

These families do use it: YOLOv1 through YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet and SSD.

Two prediction-time options make `iou` matter even for a set predictor, because
both merge boxes after the model has finished:

- `tiling=True` reconciles overlapping tiles with per-class NMS at `iou`
- `augment=True` merges flipped views with per-class NMS at `iou`

Both are covered in [Inference performance](/docs/predict/performance).

Open-vocabulary detectors have their own rule. A family whose processor runs
NMS declares its own default threshold and honors `iou`, which is the case for
OMDet-Turbo. Families that suppress nothing, Grounding DINO, OWLv2 and OV-DEIM,
emit a warning when `iou` is passed. That warning is the only one of its kind in
the library.

## max_det

`max_det` caps how many predictions come back for one image. It applies
everywhere, but through different mechanisms: an NMS family truncates after
suppression, a set predictor uses it as the size of its top-k selection.

Some families clamp below whatever you ask for, because their upstream
reference configuration does. SSD caps at 200, RTMDet instance segmentation at
100, and FCOS at its own per-image detection limit. Raising `max_det` past
those has no effect.

The one place `max_det` is applied centrally rather than per family is tiled
inference, where the merged list is truncated after tiles are reconciled.

## Class filtering

<code-tabs name="classes" />

`classes` takes a list of class ids and keeps only predictions whose class is in
it. Ids index `result.names`, and the surest way to get one is to read `names`
off a result rather than assuming a dataset ordering.

Filtering happens centrally, after each family's postprocessing, in the single
funnel every prediction path goes through. That has two consequences worth
knowing. It works on every family, including the ones with no NMS. And it also
filters the payloads aligned with the boxes, so masks, keypoints and oriented
boxes are cut down alongside them rather than left mismatched.

On the command line, `classes` accepts a bare integer, a list, or a
comma-separated string:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Filtering is not free accuracy. A model still spends its budget predicting
classes you then discard, and `max_det` is applied by the family before the
filter, so an image crowded with unwanted classes can hit the cap before your
class is reached. Lower `conf` or raise `max_det` if that happens.

## agnostic_nms

`agnostic_nms` is accepted and does nothing. Passing it raises a warning saying
it is a no-op for command line compatibility, and the argument is discarded.

There is no class-agnostic suppression mode. Every NMS call in the library is
class-aware, so two overlapping boxes of different classes both survive, at any
`iou`. Where that is a problem, filter with `classes` first, or suppress across
classes yourself on `result.boxes`.

## What predict rejects

Two arguments raise instead of warning: `visualize` and `embed` both raise
`NotImplementedError`. For embeddings, load the model with `task="embed"` and
call `predict` or `embed` normally.

Anything unrecognized raises `TypeError` naming the supported options, so a
typo fails immediately rather than being silently ignored.

These are accepted, warned about and discarded: `agnostic_nms`, `boxes`, `dnn`,
`half`, `line_width`, `retina_masks`, `show_conf`, `show_labels` and `verbose`.
