---
title: Full export matrix
seo_title: "LibreYOLO export support matrix and its rules"
description: "How LibreYOLO decides whether a family, task and format combination exports: the twelve formats, the three tiers, the fallback rules and the parity thresholds."
lead: "Export support is a lookup on the triple (family, task, format). This page describes the shape of that matrix, the rules that fill the cells no explicit entry covers, and how to query it for a combination you care about."
keywords:
  - libreyolo export support
  - export matrix
  - onnx tensorrt openvino tflite
  - libreyolo formats command
  - export parity threshold
  - NotImplementedError export
last_verified: "1.5.0"
verification: "Formats, tiers, fallback order, task and family blocks and NCNN blocks read from libreyolo/export/support.py; aliases and shared arguments from libreyolo/export/exporter.py; tier definitions from docs/adr/0011-export-support-tiers.md; parity thresholds from docs/export_support.md, all at v1.5.0. Per-combination cells are not transcribed here; query them with the snippet below."
snippets:
  usage:
    - label: Query the matrix, no model needed
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Export, and read a rejection
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # Check before calling: a blocked combination raises in preflight
        # and the message carries this reason.
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
---

## Shape of the matrix

The matrix is keyed by `(family, task, format)`. Family keys are the canonical
names from the model registry, task keys come from `libreyolo.tasks.TASKS`,
and there are twelve formats:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` additionally accepts two aliases: `engine` for
`tensorrt`, and `litert` for `tflite`, which is the current name for TensorFlow
Lite. The format and the `.tflite` suffix are unchanged.

<code-tabs name="usage" />

Because a cell is a function of three keys, the full grid is large and changes
every release. It is generated rather than written by hand, and lives in
`docs/export_support.md` in the library repository. Query the matrix from
Python or the CLI rather than reading a copy.

## The three tiers

| Tier | Meaning |
|---|---|
| `validated` | Numeric parity is covered in CI or a documented nightly run |
| `available` | Conversion is implemented, but numeric runtime parity evidence has not been recorded |
| `blocked` | Preflight raises `NotImplementedError` with a reason before tracing |

Validated and available combinations both proceed without an acknowledgement
or a blanket warning. Their recorded evidence and constraints stay visible in
the generated documentation. A blocked combination fails before dependency
checks, calibration loading, tracing or artifact creation.

Adding a validated entry requires a parity test and a `since` field.

A `SupportEntry` carries four fields: `tier`, a `reason` string, the `since`
release, and a `constraint` string. The constraint is the part that matters at
integration time: a check mark applies only under the conditions it names,
which are typically a fixed input canvas, batch 1, FP32, and a named runtime
version.

## How a cell is decided

`get_support(family, task, fmt)` resolves in this order. The first rule that
matches wins.

1. An unknown task, or a format outside the twelve, returns `blocked`.
2. An explicit `(family, task, format)` entry returns as recorded.
3. A family-wide block returns `blocked` with that family's reason.
4. A task-wide block returns `blocked` with that task's reason.
5. For `ncnn`, a family on the NCNN block list returns `blocked`.
6. `mnn` returns `blocked`: no runtime contract for this family and task.
7. `rknn` returns `blocked`. RKNN in this version is limited to the exact simulator-tested detection variants: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s and PicoDet-s on RK3588.
8. `tensorrt` and `openvino` return `available`: the converter path exists but runtime parity has not been recorded for that family and task.
9. `tflite`, `paddle`, `coreai` and `coreml` return `blocked`, each with its own reason.
10. Everything else returns `available`: conversion is implemented, numeric runtime parity is not recorded.

The asymmetry in steps 8 through 10 is deliberate. TensorRT and OpenVINO
convert generically from ONNX, so an unlisted combination is worth attempting.
TFLite, Paddle, Core AI and CoreML each need a per-family path, so an unlisted
combination is a rejection rather than an invitation.

## Blocked tasks

These tasks are blocked for any family with no explicit entry.

| Task | Reason |
|---|---|
| `ocr` | Two networks with dynamic per-region cropping do not fit the single-graph export contract |
| `point` | The family is not wired to the shared point heatmap and backend peak-decoding contract |
| `semantic` | The family is not wired to the shared dense-logits and backend argmax contract |
| `mesh` | Body-mesh graph outputs, metadata and runtime contract are not defined |
| `normal` | The family is not wired to the fixed-canvas dense unit-normal and backend renormalization contract |
| `panoptic` | Panoptic export has no backend runtime contract |
| `gaze` | The family is not wired to the shared two-head logits and backend expectation-decoding contract |

An explicit entry overrides these, which is how, for example, a wired
semantic family still exports.

## Blocked families

| Family | Blocked for |
|---|---|
| `depth_anything3` | Every format; its depth graph is not in the exported-runtime contract |
| `domedetr` | Every format. PAQI sets the query count per image, so a traced graph is valid only for the image it was traced on. Use D-FINE for an exportable DETR |
| `eomt` | Instance and panoptic export, which have no runtime parsing |
| `l2cs` | Anything outside ONNX, TorchScript, ExecuTorch, TensorRT and OpenVINO |
| `hrnet` | Anything outside ONNX, TorchScript, OpenVINO and TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Every format; promptable model export is out of scope for the v1 runtime contract |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Every format; open-vocabulary runtime export is out of scope for v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Every format; generative VLM export is out of scope for v1 |

PicoSAM3 is the exception in the promptable tier: it exports its raw 96 pixel
ROI network to ONNX.

## Blocked for NCNN

DETR-style decoders need sampling operations NCNN does not implement, so these
families are blocked for `ncnn` unless an explicit entry says otherwise:
Deformable DETR, DETR, DINO-DETR, D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR,
RT-DETRv2, RT-DETRv4, RF-DETR and EC. The rejection names ONNX, OpenVINO,
TorchScript and TensorRT as the alternatives.

## Parity thresholds

A validated cell means the exported artifact reproduced the native model
within these bounds:

| Task group | Threshold |
|---|---|
| Detection and OBB | Matched box IoU above 0.95, score MAE below 0.01 |
| Segmentation and panoptic | Mask IoU above 0.95 |
| Pose | Keypoint L2 below 2 pixels at native resolution |
| Classification | Logits cosine above 0.999 and equal top-1 class |
| Depth and restoration | PSNR above 40 dB against native output |
| Surface normals | Mean angular error below 0.1 degree |
| Point | Peak locations equal within one output cell |

DETR query rows are an unordered set, so DETR-family parity aligns query rows
as a set rather than positionally.

## Exporting

<code-tabs name="export" />

A blocked combination raises `NotImplementedError` in preflight and the
message carries the recorded reason. `validated_alternatives(family, task)`
returns the formats that are validated for that pair, which is the useful
thing to print next to a rejection.

The arguments every exporter shares are listed on the
[model API page](/docs/reference/model-api). Format-specific arguments live on
the individual format pages.

## Reading a constraint

A validated cell is a claim about one measured configuration, not about the
format in general. A constraint string such as
`FP32, batch 1, fixed 520x520 input` means parity was recorded at that shape
and precision. Exporting at a different resolution or batch size still
produces an artifact; it just is not the configuration the number came from.
