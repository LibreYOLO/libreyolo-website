---
title: Ensemble API
seo_title: "LibreEnsemble API and fusion operations"
description: "LibreEnsemble, ExternalDetector, and the three fusion ops in libreyolo.ops: weighted boxes fusion, its seeded variant, and class-aware NMS fusion."
lead: "LibreEnsemble runs several detectors on the same image and fuses their detections into one Results. Fusion happens after each member's own postprocessing, so members keep their own input size, normalization and suppression."
keywords:
  - LibreEnsemble
  - weighted boxes fusion
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - min_votes consensus
last_verified: "1.5.0"
verification: "Signatures and defaults read from libreyolo/ensemble/model.py and libreyolo/ops/fusion.py at v1.5.0. Design intent from docs/adr/0004-model-ensembling.md."
snippets:
  usage:
    - label: Two members, default fusion
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])
        result = ens(SAMPLE_IMAGE, conf=0.25)[0]

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Consensus and per-member thresholds
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])[0]
        print(len(result))
  ops:
    - label: Fusion op, no model involved
      language: python
      code: |
        import torch
        from libreyolo.ops import weighted_boxes_fusion

        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0, 49.0]])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])
        model_ids = torch.tensor([0, 1])

        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )
        print(fused)
---

## LibreEnsemble

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

| Argument | Default | Meaning |
|---|---|---|
| `members` | | Two or more detectors |
| `weights` | `None` | Per-member trust factors; all `1.0` when omitted |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"`, or a callable |
| `fusion_iou` | `0.55` | IoU threshold for fusion clustering |
| `min_votes` | `1` | Keep only boxes confirmed by at least this many members |

A member is a weights path resolved through the `LibreYOLO()` factory, an
already-constructed model, an exported backend, or an `ExternalDetector`.
Every member must be a detect-task model.

<code-tabs name="usage" />

Construction rejects fewer than two members, a `weights` list of the wrong
length, a non-positive weight, a `min_votes` that is not a positive integer,
and a `min_votes` larger than the member count. `fusion="nms"` with
`min_votes > 1` also raises, because NMS discards cluster membership and
cannot count votes.

`weights` scales the trust placed in each member. Higher weight pulls fused
coordinates and scores toward that member. The convention is to make them
proportional to validation mAP.

## Class spaces

Members with identical `names` pass straight through. Otherwise the class
spaces are unioned by name, member class IDs are remapped through lookup
tables, and the fused `Results.names` is the union. Fusion merges boxes only
within the same unified class, so a class only one member knows passes through
unfused. A mismatch logs a warning at construction.

`min_votes` is capped per class by how many members' label spaces contain that
class, so consensus stays meaningful on partially shared vocabularies.

## Calling the ensemble

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` is an alias for `__call__`. The return is the usual `Results`, whose
`speed` breaks the cost down per member and adds a `fusion` entry.

`conf`, `iou` and `device` broadcast to every member and also accept one value
per member, so `conf=[0.25, 0.4]` gives member 0 a threshold of 0.25 and
member 1 a threshold of 0.4. `imgsz` broadcasts when it is an int or a tuple
and is per-member only when it is a list, so `imgsz=(480, 640)` is one
rectangular size for everyone while `imgsz=[480, 640]` is 480 for member 0 and
640 for member 1. Each entry must be valid for that member's family.

`augment` broadcasts to members that support test-time augmentation, and
exported backends ignore it. `classes` takes union class IDs and `max_det`
applies to the fused result, so members run generously and the ensemble trims
once. `batch` is accepted for API parity; images are processed sequentially.

`val()` and `export()` raise `NotImplementedError`. Validate and export the
members individually.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Adapts any detection callable into a member. `fn` takes a PIL image and
returns `(boxes, scores, labels)`, where boxes are xyxy in original-image
pixels and labels are class IDs valid in `names`. Tensors, arrays and nested
lists all work. LibreYOLO imports nothing from the external code.

The adapter validates the return: it must be a 3-tuple, boxes must have shape
`(N, 4)`, the three arrays must be the same length, and every class ID must
appear in `names`. Detections at or below `conf` are dropped before fusion.

## Fusion operations

The fusion primitives are standalone torch ops in `libreyolo.ops`. They are
model-free and importable on their own, which is why they are exported
separately from the ensemble.

<code-tabs name="ops" />

All three take the same positional arguments, `boxes, scores, labels,
model_ids`, and return `(boxes, scores, labels)`.

| Op | Registry key | Behavior |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Sequential, paper-faithful weighted boxes fusion |
| `wbf_seeded` | `wbf_seeded` | Parallel one-pass variant of the same reduction |
| `nms_fusion` | `nms` | Concatenate everything and apply class-aware NMS |

`FUSIONS` maps the three registry keys to the callables, and `LibreEnsemble`
looks up `fusion=` there.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` takes the identical signature. `nms_fusion` takes the same
arguments except `conf_type`, and raises `ValueError` when `min_votes > 1`.

In `weighted_boxes_fusion`, detections are visited in order of decreasing
weight-scaled confidence. Each one either joins the existing cluster whose
running fused box it overlaps best, at IoU above `iou_thr` and with the same
label, or starts a new cluster. A cluster's fused box is the
confidence-weighted average of its members' coordinates, and its score is the
weighted mean or maximum of their confidences, rescaled so that boxes
confirmed by fewer models score lower.

`wbf_seeded` picks cluster seeds with class-aware NMS at `iou_thr`, assigns
every detection to its best-IoU seed of the same label, then reduces each
cluster the same way. Cluster shapes never shift mid-pass, so the whole op is
fixed-shape tensor math. The two variants agree whenever clusters are
unambiguous and can differ slightly on overlapping cluster chains.

`nms_fusion` keeps the highest-confidence box of each overlapping group,
unchanged. Per-model `weights` scale confidences for the suppression ranking
only, and surviving boxes keep their original scores.

## Custom fusion

`fusion=` also accepts a callable with the same signature as the ops above.
Its name is recorded on `ens.fusion`, or `"custom"` when it has none. The
return is validated: it must be a `(boxes, scores, labels)` triple with
consistent shapes.
