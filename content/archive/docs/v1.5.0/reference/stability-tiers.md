---
title: Stability tiers
seo_title: "What each LibreYOLO support tier means"
description: "The tier vocabulary LibreYOLO uses: the three export support tiers, the four API tiers, the six coverage groups, and what none of them promise."
lead: "LibreYOLO uses the word tier for three separate things: the evidence behind an export path, the call contract a model family answers to, and the coverage group a family is enrolled in. This page defines each one and says what it does not imply."
keywords:
  - libreyolo support tier
  - validated available blocked
  - export support tiers
  - libreyolo coverage groups
  - g0 g1 g2 g3 g4
  - model tiers
last_verified: "1.5.0"
verification: "Export tiers from docs/adr/0011-export-support-tiers.md and libreyolo/export/support.py; coverage groups and per-family counts from libreyolo/models/registry.py MODEL_GROUPS; the from-scratch gate from libreyolo/models/base/model.py and libreyolo/cli/commands/train.py; the CLI inventory read from libreyolo/models/inventory.py; API tiers from libreyolo/models/sam/, openvocab/ and vlm/ package docstrings and base.py contracts, all at v1.5.0. The reader-facing group labels (Flagship, Core, Supported, Inference only, Museum, Sibling tier) are the site's own vocabulary for the same groups, from src/data/docs/registry.json."
snippets:
  usage:
    - label: Read both classifications for one family
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
---

## Export support tiers

The tier that decides whether a call succeeds. It applies to the triple
`(family, task, format)`, and every combination has exactly one.

| Tier | Meaning | What happens on `export()` |
|---|---|---|
| `validated` | Numeric parity is covered in CI or a documented nightly run | Runs |
| `available` | Conversion is implemented, but numeric runtime parity evidence has not been recorded | Runs |
| `blocked` | No supported path | Raises `NotImplementedError` in preflight, with the reason |

Validated and available both proceed without an acknowledgement or a blanket
warning. The difference is evidence, not permission: a validated entry has a
parity test behind it and a `since` release, and an available entry does not
yet. A CoreML conversion without a macOS prediction run, for example, is
available and not validated.

A blocked combination fails before dependency checks, calibration loading,
tracing or artifact creation, so nothing partial is written.

Every validated cell carries a constraint describing the configuration the
parity number came from, typically a fixed input canvas, batch 1, FP32 and a
named runtime version. Read it as a claim about that configuration rather than
about the format in general. The rules that fill cells with no explicit entry
are on the [export matrix](/docs/reference/export-matrix) page.

<code-tabs name="usage" />

## API tiers

The tier that decides what a call looks like. A family sits in exactly one,
chosen by call contract rather than by architecture.

| Tier | Factory | Contract |
|---|---|---|
| Detector factory | `LibreYOLO` | One promptless forward returns every object it found, with calibrated scores. Members register themselves by recognizing a checkpoint |
| Promptable segmentation | `LibreSAM` | A forward is meaningless without a per-image spatial or concept prompt supplied at call time. Interactive and stateful: encode once, prompt many times |
| Open-vocabulary detection | `LibreOpenVocab` | Text-conditioned discriminative detectors. The class list is a prompt, set by `set_classes` |
| Vision-language | `LibreVLM` | A generative model driven as a detector. The class list is a prompt and the confidence is a placeholder |

The three sibling tiers deliberately do not register into the detector
factory, which is why `LibreYOLO("some-alias")` does not reach them. They load
by size alias and autodownload rather than by checkpoint sniffing.

All four return the same `Results`, so downstream code is unchanged across
them. What differs is which methods work: the sibling tiers raise
`NotImplementedError` for `train()`, `val()` and `export()`, and the SAM and
open-vocabulary tiers raise for `track()` as well. Each tier page lists its
own exclusions.

## Coverage groups

The classification that decides which families a cross-family test run
includes, and the one a reader is most likely to meet on a model page. Every
registered family is enrolled in exactly one group, and a test fails when a
registered family is missing from the enrollment. `GROUPS` in
`libreyolo/models/registry.py` is the source of the Meaning column below;
`MODEL_GROUPS` in the same file assigns every family, and the Families column
counts that assignment directly. The Label column is the shorter name the
site uses for the same group on a model page header.

| Group | Label | Families | Meaning |
|---|---|---|---|
| `g0` | Flagship | 2 | Flagship anchors required in shared-feature coverage |
| `g1` | Core | 10 | Trainable detector coverage set |
| `g2` | Supported | 14 | Additional trainable-family coverage set |
| `g3` | Inference only | 35 | Families without a training implementation |
| `g4` | Museum | 5 | Historical families with inference coverage |
| `s` | Sibling tier | 21 | Sibling APIs (SAM, open-vocab, VLM, zero-shot) covered separately |

That is 87 families across six groups. `g3` alone holds more families than
every other group combined, because most of the registry is inference-only
lineage and museum coverage rather than actively trained detectors.

For a reader choosing a model, the group says where to expect engineering
attention, not how accurate a family is. `g0` and `g1` are where a new
feature is designed and land first; `g2` is kept green in CI but a feature
lands there opportunistically rather than on the same release wave. `g3`
states an absence rather than a limit: predict, validate and, where the
family supports it, export all still work, and `train()` on a `g3` or `g4`
family raises `NotImplementedError` naming the reason rather than doing
something silently partial. `s` families do not sit in this trade-off at
all, because they load through their own factory rather than `LibreYOLO()`.
See [core concepts](/docs/concepts) for how a group fits alongside task,
family and size when reading a checkpoint filename.

A group does not grant or restrict a user-facing capability by itself.
Support comes from the family's implemented API and from format-specific
capability checks, never from group membership alone. Groups classify
families, not tasks, so a task-scoped coverage run names the task explicitly,
as in "g1 detect".

Two places read the group at runtime rather than only in tests.
`collect_model_inventory()` in `libreyolo/models/inventory.py` attaches the
group to every entry the CLI inventory prints, and `pretrained=False`
triggers the special from-scratch reinitialization path only for families in
`g0` and `g1`. Outside those two groups the check in
`libreyolo/models/base/model.py` is skipped entirely, so `pretrained=False`
reaches the family's own `train()` as an ordinary keyword instead.

## Training

A family in `g3` or `g4` has no training implementation, and calling `train()`
on one raises. That is a property of the family's code, not of its group: the
group records the fact rather than causing it.

For a family that does train, whether an individual augmentation knob reaches
the pipeline is a separate question with its own three-value vocabulary,
`used`, `gated_by_mosaic` and `ignored`. See the
[augmentation matrix](/docs/reference/augmentation-matrix).

## What a tier does not tell you

A tier is not an accuracy claim. A validated export says the artifact
reproduces the native model within a stated threshold; it says nothing about
how well the native model scores on a dataset. Benchmark numbers live on the
model pages.

A tier is also not a licensing statement. Weight licenses vary within a family
and the repository hosting a specific checkpoint is authoritative. A family
being in the detector factory says nothing about whether its published weights
permit commercial use.
