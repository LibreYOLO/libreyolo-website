---
title: Ensembling detectors
seo_title: "Ensembling detectors in LibreYOLO"
description: "Run several detectors on one image and fuse their boxes with weighted boxes fusion or NMS, including models with different class lists."
lead: "LibreEnsemble runs two or more detectors over the same decoded image and fuses their boxes into one Results object. Members keep their own weights, thresholds, devices and class lists."
keywords:
  - model ensemble object detection
  - weighted boxes fusion
  - wbf python
  - combine two detectors
  - fuse bounding boxes
  - LibreEnsemble
  - ensemble detection python
  - min_votes
last_verified: "1.5.0"
verification: "Constructor and call signatures, defaults, validation errors, class-space unification, vote counting and the returned Results read from libreyolo/ensemble/model.py. Fusion algorithms and their arguments from libreyolo/ops/fusion.py. Design intent from docs/adr/0004-model-ensembling.md. Usage patterns cross-checked against tests/unit/test_ensemble.py and tests/unit/test_ops_fusion.py."
snippets:
  basic:
    - label: Two detectors, fused
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # Members can be checkpoint paths or already-loaded models.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Weights and a vote requirement
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # by convention, proportional to validation mAP
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # keep only boxes both members found
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Per-member thresholds
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # A scalar applies to every member; a list is read per member.
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: Bringing in a detector LibreYOLO did not load
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Return (boxes, scores, labels): xyxy in original-image pixels.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: The same sources a single model takes
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Replace clip.mp4 with a video file on disk.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
---

## What an ensemble is

`LibreEnsemble` takes two or more detectors, runs each on the same image, and
fuses their boxes into a single `Results`. It is a prediction-time construct:
there is nothing to train, and the members stay independent models that can be
validated and exported on their own.

Detection is the only task it supports. A member whose task is anything else
raises `ValueError` at construction, naming the member index and its task.

Both names are imported lazily, so they cost nothing until used:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Building one

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` is a sequence of two or more. A `str` or `Path` entry is loaded
through `LibreYOLO()`; anything else has to be callable and expose a `names`
dict. Fewer than two raises `ValueError`, and passing a bare string raises
`TypeError` rather than iterating its characters.

`weights` defaults to `None`, which is uniform weighting. Supplied weights must
be one per member and strictly positive, so a zero weight raises rather than
silently dropping a member. The documented convention is to set them
proportional to each member's validation mAP.

`fusion_iou` defaults to `0.55` and is the IoU at which boxes from different
members are clustered together. It is a different threshold from the per-call
`iou`, which is each member's own NMS setting.

`min_votes` defaults to `1`, meaning any single member can carry a box. Raising
it keeps only clusters confirmed by that many distinct members. It must be a
positive integer no larger than the member count, and it is capped per class to
the number of members that actually know that class, so a class only one member
was trained on is not silently erased.

## Fusion methods

Three are accepted by name, and a callable is accepted too.

| `fusion` | Behavior |
|---|---|
| `"wbf"` | Weighted boxes fusion, sequential and faithful to the paper. The default |
| `"wbf_seeded"` | One-pass weighted boxes fusion; class-aware NMS picks cluster seeds |
| `"nms"` | Concatenate every member's boxes, then class-aware NMS |

Weighted boxes fusion averages the coordinates of a cluster weighted by
confidence, producing a box no single member proposed. The two weighted
variants agree whenever clusters are unambiguous and can differ slightly on
chains of overlapping clusters. `"nms"` picks a survivor instead of averaging,
so survivors keep their original scores, and weights only influence which box
wins. Because it selects rather than clusters, it cannot count votes: combining
`fusion="nms"` with `min_votes` greater than `1` raises `ValueError`.

Weighted boxes fusion rescales a cluster's score by the share of member weight
that backed it. With two equally weighted members, a box only one of them found
keeps half its score: `0.9` becomes `0.45`. A fused confidence can therefore
fall below the `conf` each member was run at, so filter on the fused score
rather than assuming the member threshold still holds.

## Members with different class lists

Members do not have to share a class list. Their label spaces are unioned by
name, and each member gets a lookup table remapping its own class ids into the
union. `ensemble.names` is that union, and it is what the returned `Results`
carries.

Boxes only ever fuse within the same class name. A class that only one member
knows passes through unfused, and it is not penalized for it: the score rescale
uses a per-class denominator, so a solo-known class keeps its score.

Partial overlap logs a warning naming the classes that are not shared by every
member. That warning is the thing to read carefully, because a checkpoint whose
class names are placeholders such as `class_0` builds a union that is disjoint
from every other member, and no cross-member fusion happens at all.

A member returning a class id outside its own `names` raises `RuntimeError`.

## Foreign detectors

<code-tabs name="external" />

`ExternalDetector(fn, names)` wraps any callable that takes a PIL image and
returns `(boxes, scores, labels)`, with boxes as xyxy in original-image pixels.
It validates arity, box shape, length agreement and that every class id appears
in `names`, and it applies the `conf` threshold itself.

This is how a detector LibreYOLO did not load takes part in a fusion.

## Calling it

<code-tabs name="sources" />

The call signature mirrors a single model's, and it accepts the same sources:
images, folders, lists, video, screen capture, webcams and network streams.
Live sources require `stream=True` for the same reason they do elsewhere.

| Argument | Default | Notes |
|---|---|---|
| `conf` | `0.25` | Per member; scalar broadcasts, or one per member |
| `iou` | `0.45` | Each member's own NMS threshold, not the fusion threshold |
| `imgsz` | `None` | A `list` is read per member; an `int` or tuple broadcasts |
| `device` | `None` | Scalar or one per member, so members can sit on different devices |
| `classes` | `None` | Filters the fused result, on union class ids |
| `max_det` | `300` | Applies to the fused result |

Because a `list` means per member for `imgsz`, `imgsz=[480, 640]` is 480 for the
first member and 640 for the second, while `imgsz=(480, 640)` is one rectangular
size for everyone. That distinction is easy to trip over.

Members are called with a `max_det` of at least 300 regardless of what you ask
for, so each runs generously and the ensemble trims once at the end.

The image is decoded once and the same object is handed to every member.
`batch` is accepted for parity and ignored; images are processed sequentially.

## What comes back

An ordinary `Results`, the same type a single model returns, with `names` set to
the union class space. Everything on
[Working with results](/docs/predict/results) applies unchanged.

The one difference is `result.speed`, which an ensemble does populate. Its keys
are `member_0`, `member_1` and so on, plus `fusion`, in milliseconds. This is
the one place in the library where `speed` is filled in.

Rows carrying non-finite boxes or scores are dropped before fusion. When members
sit on different devices, fusion runs on the device of the first member that
returned anything.

## What an ensemble cannot do

`val()` and `export()` both raise `NotImplementedError` and point you at the
members: validate and export each one individually. There is no `train` method
at all, so calling it raises `AttributeError`.

Half precision is not handled at the ensemble level. `half=True` hits the same
warned no-op path it does everywhere else; configure precision on each member.

There is no command line interface for ensembling. It is a Python API.
