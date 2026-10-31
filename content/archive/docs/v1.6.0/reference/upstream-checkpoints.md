---
title: Upstream checkpoints
seo_title: "Loading upstream checkpoints in LibreYOLO"
description: "How auto-conversion turns a released upstream checkpoint into a LibreYOLO v1.0 one: the layouts it unwraps, which families recognize what, and where it stops."
lead: "LibreYOLO families are ported from upstream projects whose released checkpoints are almost loadable but carry no LibreYOLO metadata. Auto-conversion recognizes those files, wraps them in schema v1.0, and writes the result beside the source."
keywords:
  - libreyolo autoconvert
  - load upstream checkpoint
  - convert_upstream_state_dict
  - upstream weights libreyolo
  - checkpoint conversion
last_verified: "1.5.0"
verification: "Behavior read from libreyolo/models/autoconvert.py and BaseModel.convert_upstream_state_dict; per-family recognizers checked by reading each family's convert_upstream_state_dict override, all at v1.5.0. RF-DETR COCO rules from docs/checkpoint_schema.md."
snippets:
  usage:
    - label: Just pass the file to the factory
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A recognized upstream file is converted on load, and the converted
        # checkpoint is written next to it.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Any LibreYOLO checkpoint loads unchanged.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
---

## What happens on load

When `LibreYOLO()` meets a `.pt` file that is not already a complete v1.0
checkpoint, it calls the auto-converter, which:

1. unwraps the tensor dict from the common upstream layouts;
2. asks every registered family whether it recognizes the layout, remapping
   keys where the upstream naming differs from the native port;
3. wraps the winner in a strict v1.0 metadata checkpoint, reading size, task
   and class count from the tensors themselves so fine-tuned checkpoints
   convert correctly;
4. writes it beside the source as `<source>-<Prefix><size>[-task].pt` and
   returns that path, so the factory loads it normally.

Nothing is asked of the caller. A file that no family claims returns nothing
and the factory reports that it could not load it.

<code-tabs name="usage" />

## Layouts it unwraps

The tensor dict is looked for in this preference order, EMA first, and each
candidate is tried until one actually holds tensors. An empty or
metadata-only EMA block therefore does not mask valid weights underneath.

| Key | Note |
|---|---|
| `ema.module` | The common EMA wrapper |
| `ema` | Legacy flat EMA wrappers that store tensors directly |
| `ema_state_dict` | Entries under a `module.` prefix are stripped |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| The file itself | A plain state dict |

Each candidate is then narrowed to its tensor-valued entries and normalized:
a leading `module.` or `_orig_mod.` prefix is stripped, and a dict whose keys
all start with `model.model.` has that prefix removed.

## Which families recognize what

Recognition is a per-family classmethod. The default implementation claims a
layout whose keys already match the native port. A family whose upstream key
naming differs overrides it with a remap, and returns nothing for layouts it
does not recognize.

Families that ship a remapping recognizer: `centernet`, `deeplabv3`,
`deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`, `yolo9`,
`yolo9_e2e`, `yolo9_p2`.

Families that decline auto-conversion outright: `efficientdet`, `eomt` and
`pidnet` return nothing from the recognizer, so their upstream files go
through a conversion script instead. `l2cs` is excluded from the generic
recognizer because it is inference-only with redistribution-restricted
weights.

RF-DETR keeps its own recognizer, because it needs the whole checkpoint rather
than just the tensor dict to detect the size and to remap COCO classes. It is
registered only when its optional dependencies are installed.

Every other registered family uses the default: it claims the file when its
own loader already recognizes those keys.

## Which family wins

Several families can claim the same file, so the resolution mirrors the
factory's dispatch rules.

A subclass claim beats its base class. Registration order follows class
creation, so a derived family registers after the base it refines, and its
positive markers must not lose to the base's broader passthrough.

Registry order then decides, because it encodes specificity: the earliest
claim is the most specific match.

The one tie registry order cannot break is DEIM against D-FINE, whose
architecture keys are identical. There, and only there, the filename is the
deciding signal, and a file whose name gives no hint is refused rather than
guessed. The filename is deliberately not consulted anywhere else, so a broad
false-positive claim can never be promoted over a more specific one purely by
what the file is called.

## Safe loading

Upstream files are loaded through the weights-only unpickler. Some upstream
training checkpoints embed library objects that unpickler rejects. Those
objects are training metadata rather than weights, so each blocked global is
retried with an inert stand-in class that satisfies the unpickler without
executing anything. The captured name is used only as a string label, never
imported, evaluated or called.

Sensitive module names are refused outright and never stubbed: `builtins`,
`os`, `sys`, `posix`, `nt` and `subprocess`. The retry loop is bounded at 32
attempts, so a file engineered to introduce an unbounded series of distinct
globals fails closed instead of spinning. Only tensors survive into the
converted checkpoint.

## Where the converted file goes

The output is written beside the source, named
`<source>-<Prefix><size>[-task].pt`. It is always rewritten rather than
reused, which keeps repeated loads of the same source fresh while avoiding
collisions with official weights or with another fine-tune of the same family,
size and task in the same directory.

When the source directory is read-only, the conversion falls back to a fresh
private temporary directory created per call, and the log line names the path
it used. Only if that also fails is the conversion dropped, with a warning.

## Existing LibreYOLO checkpoints

A file carrying a LibreYOLO-specific marker, `libreyolo_version` or
`model_family`, belongs to the normal load path and is not re-converted. The
skip applies only to a passthrough claim, meaning one where the keyset was
unchanged. A claim whose conversion changed the keyset is proof of a foreign
upstream layout and is accepted even on a marked file.

`schema_version` is deliberately not treated as a marker, because other
training and export tools use that generic name, and neither are `names`,
`nc`, `size`, `task` or `imgsz`, because an upstream fine-tune may carry them
too. A foreign fine-tune that merely carries a generic `names` key is
therefore not marked, so its native-keyed claim converts normally and derives
the class count from the tensor head rather than being mis-loaded as 80 class.

## Metadata the converter reads

Class names are taken from a top-level `names` key, or from `class_names`
inside an `args` or `hyper_parameters` block. A names map keyed by labels
rather than by class index is unusable and is replaced by generated defaults.
A names list longer than the detected class count is trimmed, because
out-of-range indices would fail the strict validator and silently abort the
conversion.

Upstream `args` are carried over as plain metadata, with any value that is not
a string, number, boolean, list or dict dropped, so nothing unsafe reaches the
saved file.

## RF-DETR COCO normalization

Upstream RF-DETR checkpoints expose a 91-output classification head, which is
COCO's 90 classes plus background. Auto-conversion normalizes a COCO RF-DETR
to the COCO-80 convention, with the remap applied at postprocessing.

A checkpoint is treated as COCO when it carries exactly 80 names, or declares
a class count of 80, or has a `coco` dataset hint, or has no class or dataset
metadata at all. That last case matters: a bare upstream state dict is the
canonical COCO-pretrained checkpoint, and it is the only metadata-less
91-output RF-DETR in distribution.

A genuine custom 90-class RF-DETR is preserved as 90 classes. It is identified
by a names list, an explicit non-80 class count, or a non-COCO dataset hint,
so the bare-checkpoint fallback does not fire for it. Empty placeholders are
ignored when deciding whether a dataset hint is present.

## Limits

Auto-conversion recognizes released upstream layouts. It does not rewrite an
architecture, and it does not make an unported model loadable. When no family
claims a file, the answer is a conversion script rather than a factory
argument: the repository ships `weights/convert_*.py` for the families that
need one, including EoMT, PIDNet and EfficientDet.

Conversion also does not invent metadata it cannot read. Size, task and class
count come from the tensors; names come from the file when present, and are
generated as `class_i` when not.
