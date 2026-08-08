---
title: Open-vocabulary API
seo_title: "LibreOpenVocab API: aliases and arguments"
description: "The LibreOpenVocab factory, its four families and every alias, set_classes, the per-family conf defaults, and the text_threshold and iou rules."
lead: "LibreOpenVocab is the factory for text-conditioned detectors. The class list is a prompt rather than a fixed head, so the vocabulary is set by set_classes and the model returns ordinary detection Results against it."
keywords:
  - LibreOpenVocab
  - open vocabulary detection
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: "1.5.0"
verification: "Aliases read from libreyolo/models/openvocab/__init__.py; repositories, sizes and thresholds from grounding_dino.py, owlv2.py, omdet_turbo.py and ov_deim.py; call rules from libreyolo/models/openvocab/base.py, all at v1.5.0. Design intent from docs/adr/0008-open-vocab-detector-contract.md."
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)[0]
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
---

## Install

The tier needs the `openvocab` extra.

<code-tabs name="install" />

## The factory

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` is an alias, not a path. Underscores fold to hyphens before lookup, so
the family-qualified names the CLI inventory prints, such as `omdet_turbo-t`
and `grounding_dino-t`, load as given. An unknown alias raises `ValueError`
listing every known alias.

The constructor accepts `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` and `text_threshold=None`. Passing `names` is
the same as calling `set_classes` right after loading. Passing
`text_threshold` to a family that does not support it raises `TypeError`.

<code-tabs name="usage" />

## Families and aliases

| Family | Aliases | Sizes | Weights |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

The default alias is `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` and `LibreOMDetTurbo` are exported at
package level and can be constructed directly with `size=`. OV-DEIM is
reachable through the factory aliases above.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Sets the vocabulary for every later `predict()` call, and returns the model so
calls can chain. The list must be non-empty, must contain only strings, and
its entries must be unique when compared case-insensitively; blank labels are
rejected. Passing a bare string raises `TypeError`, because it would enumerate
into one-character classes.

After the call, `model.names` maps `0..N-1` to the labels in the order given,
and `model.nb_classes` is `N`.

## Call arguments

The tier reuses the standard predict surface with three differences.

`conf` defaults to the family's own value rather than the shared 0.25:

| Family | Default conf | Suppression |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Its own post-processing, threshold 0.5, honors `iou=` |
| OV-DEIM | 0.25 | One-to-one matching with top-K selection, no suppression |

`iou=` only means something for a family that runs suppression. OMDet-Turbo
takes the threshold as an argument and defaults it to 0.5 when `iou=` is
unset. The other three suppress nothing, so passing `iou=` there emits a
warning and is ignored.

`text_threshold=` is Grounding DINO only, where it defaults to 0.25. It can be
passed at construction for a persistent value, or per call. A per-call value
cannot be combined with `stream=True`, because streamed results are generated
lazily; set it on the constructor instead. Every other family raises
`TypeError` for it.

`imgsz=` raises `ValueError`: the preprocessing pipeline owns resizing for
this tier. `augment=True` raises as well, since test-time augmentation is out
of scope here. Input sizes are recorded per family for reference only:
Grounding DINO 800, OWLv2 960 and 1008, OMDet-Turbo 640, OV-DEIM 640.

## Not supported

`train()`, `val()`, `track()` and `export()` all raise
`NotImplementedError`. Fine-tune upstream and load the resulting weights;
run `predict()` per frame in place of tracking. Validation would need a
dedicated validator, because the shared detection validator calls the model
with image tensors while this tier requires text-conditioned inputs.
