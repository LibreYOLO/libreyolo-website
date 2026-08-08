---
title: Checkpoint schema
seo_title: "LibreYOLO checkpoint metadata schema v1.0"
description: "The metadata every LibreYOLO .pt checkpoint carries: required keys, per-task additions, export runtime keys, quantized manifests and training fields."
lead: "A LibreYOLO .pt file is a flat dictionary saved with torch.save. The model key holds the state dict; the other top-level keys are metadata that identifies the checkpoint without filename parsing or state-dict sniffing."
keywords:
  - libreyolo checkpoint schema
  - schema_version 1.0
  - model_family
  - libreyolo checkpoint metadata
  - quant manifest
  - wrap_libreyolo_checkpoint
last_verified: "1.5.0"
verification: "Mirrors docs/checkpoint_schema.md in the libreyolo repository at v1.5.0, cross-checked against libreyolo/utils/serialization.py and BaseModel.save."
snippets:
  usage:
    - label: Read the metadata off a checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint
        import torch

        # Download a checkpoint, then re-save it so a local path exists.
        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")

        loaded = torch.load("roundtrip.pt", map_location="cpu", weights_only=False)
        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)

        print(metadata["schema_version"], metadata["model_family"])
        print(metadata["size"], metadata["task"], metadata["nc"], metadata["imgsz"])
        print(len(state_dict), "tensors")
---

## Schema v1.0

Every official LibreYOLO `.pt` checkpoint contains:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Key | Type | Meaning |
|---|---|---|
| `model` | state dict | The model weights |
| `schema_version` | str | Metadata contract version; v1.0 uses the string `"1.0"` |
| `libreyolo_version` | str | The version that produced the checkpoint |
| `model_family` | str | A registered family, such as `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Variant within the family, such as `t`, `s`, `r18`, `atto` |
| `task` | str | Canonical task name |
| `nc` | int | Positive class count |
| `names` | dict | `dict[int, str]` with keys in `0..nc-1` |
| `imgsz` | int | Positive square input resolution, or the legacy scalar for a rectangular contract |

`task` is one of `detect`, `segment`, `semantic`, `panoptic`, `pose`,
`classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` or `mesh`.

Official checkpoints write every `names` key. Readers may pad missing keys
with `class_i` labels for legacy sparse mappings, but out-of-range keys are
invalid.

Rectangular checkpoints keep a scalar `imgsz` for legacy readers, set to
`max(imgsz_h, imgsz_w)`, and additionally write `imgsz_h` and `imgsz_w` with
the real dimensions. A reader that understands the rectangular fields must
prefer them over the scalar. Families with a fixed rectangular contract, such
as HRNet pose, reject incompatible runtime sizes.

The schema is deliberately flat, and `model` is deliberately a state dict.

<code-tabs name="usage" />

## Pose additions

Pose is usually single-class, `nc: 1` with `person`, but the YOLO-NAS pose
head also supports multi-class pose with one shared keypoint skeleton, in
which case `nc` and `names` describe the classes as in detection. Runtime pose
exports emit `scores` with shape `[batch, anchors, nc]`.

| Key | Meaning |
|---|---|
| `num_keypoints` | Positive keypoint count used by the pose head |
| `keypoint_dim` | `2` for `x,y` labels or `3` for `x,y,visibility` labels; model outputs always expose `x,y,visibility` |
| `oks_sigmas` | Optional per-keypoint OKS sigmas; the task default for `num_keypoints` is used when absent |
| `num_keypoints_per_class` | Optional per-class keypoint counts for GroupPose-style heads whose keypoint tensor is padded by class; `0` for classes without keypoints |

## Mesh additions

Mesh checkpoints use `task: "mesh"`, `nc: 1` and `names: {0: "person"}`.
Parameter layouts differ between body models, so the dimensions are recorded
rather than assumed.

| Key | Meaning |
|---|---|
| `body_model` | The parameterization, such as `mhr`; required, and used to interpret every field below |
| `num_betas` | Identity and shape coefficient count; 45 for MHR |
| `num_body_pose` | Width of the body-pose parameter block; 130 for MHR. A flat vector, not one triplet per joint, because rig joints carry different degrees of freedom |
| `num_vertices` | Vertex count the decoder emits; 18439 for MHR |
| `num_joints` | Joint count the decoder emits; 127 for MHR |
| `rotation_format` | How rotations are encoded, such as `euler_zyx` for MHR or `axis_angle`. Never inferred from tensor shape, since a 3-vector is ambiguous |

## Dense-task placeholders

Several tasks predict dense maps rather than classes, so the class-like slots
exist only for schema compatibility.

| Task | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Edge predictions are dense float32 probability maps in `[0, 1]`.

Restore checkpoints may add `degradation`, a short corruption label such as
`deblur`, `denoise` or `super-resolution`; `dataset`, a provenance label such
as `GoPro` or `SIDD`; and `scale`, a positive integer output-to-input upscale
factor, for example `4` for a x4 super-resolution model. Absent or `1` means
the restored image keeps the input resolution. The runtime also derives the
scale from the family and size, so `scale` is provenance metadata rather than
a load-time requirement.

## OCR additions

The `ppocr` family ships one composite checkpoint per tier whose `model` state
dict holds two submodels under the `det.*` and `rec.*` key namespaces.

| Key | Meaning |
|---|---|
| `charset` | The full CTC alphabet in output-index order: index 0 is the CTC blank, then the recognition dictionary, then the space character. Loaders must read it from the checkpoint, never from a side file |
| `pipeline` | Pipeline defaults baked at conversion time: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Runtime arguments may override them per call |
| `components` | Reserved for optional pipeline stages such as document orientation, unwarping and textline rotation. Empty in v1 |

## Export runtime metadata

Exported artifacts use the same rectangular dual-write convention: `imgsz_h`
and `imgsz_w` are written next to the legacy scalar `imgsz`, and a reader that
does not understand the rectangular fields must not silently treat the scalar
as a square contract.

Rectangular runtime support is family-scoped and format-scoped. YOLO9-family,
HRNet, NAFNet and Real-ESRGAN exports may use non-square `imgsz_h` and
`imgsz_w` in supported formats; families or formats without explicit
rectangular support reject the metadata rather than preprocessing those
artifacts as square. HRNet exports are fixed, batch-one, FP32 person-crop
heads, where W32 accepts 256x192 and W48 accepts 384x288, and the person
detector is not embedded in the graph.

Embedded-NMS exports may add these flat keys:

| Key | Meaning |
|---|---|
| `nms` | String boolean; `"true"` means the graph includes an embedded post-processing output |
| `nms_conf` | Confidence threshold baked into the embedded output |
| `nms_iou` | IoU threshold baked into the embedded output |
| `max_det` | Maximum post-NMS detection rows the embedded output emits |
| `nms_raw_output` | String boolean; `"true"` means the graph also exposes an auxiliary raw detector output |

For ONNX YOLO9 detection exports with `nms=true`, output `0` (named `output`)
is the standalone post-NMS tensor at the export-time thresholds. When
`nms_raw_output=true`, output `1` (named `raw`) is reserved for LibreYOLO
backends so they can apply native original-canvas clipping and runtime
`predict(conf=..., iou=..., max_det=...)` semantics. Third-party consumers
should use the first output.

Pose exports may add `num_keypoints`; `keypoint_dim`, where GroupPose-style
raw exports may use larger values such as `8` when the tensor includes
precision or class-logit fields; `num_keypoints_per_class` as a JSON-encoded
list, where zero-keypoint class slots must be preserved because they define
the schema; and `pose_input`, where `"person_crop"` means the graph consumes
one already-extracted crop and contains no detector. HRNet runtime exports
require that value.

Classification exports may add `crop_pct`, a float center-crop ratio whose
pre-crop resize target is `round(imgsz / crop_pct)` and which defaults to
`0.875` when absent, and `interpolation`, `"bilinear"` or `"bicubic"`,
defaulting to `"bilinear"`.

ExecuTorch exports write the flat metadata to a required `<program>.pte.json`
sidecar. The v1 contract is CPU, FP32, batch 1 and a fixed input canvas, and
it additionally requires `executorch_version`, `executorch_delegate` equal to
`"xnnpack"`, and a positive `executorch_delegate_partitions`. The loader
rejects a sidecar that claims another delegate, dynamic shapes, or non-FP32
precision.

MNN exports write the flat metadata to a required `<model>.mnn.json` sidecar.
The v1 contract is CPU, FP32, detection-only and a fixed NCHW input shape, and
it additionally requires `mnn_version`, `mnn_backend` equal to `"cpu"`,
ordered non-empty `mnn_input_names` and `mnn_output_names`, `mnn_input_shape`
as four positive integers in `[batch, channels, height, width]` order, and
`mnn_batch` equal to `mnn_input_shape[0]`. The loader rejects dynamic,
non-FP32, non-detection, unsupported-family or inconsistent shape metadata.

A `.pte` and a `.mnn` are backend-specific artifacts, not PyTorch
checkpoints.

## Quantized checkpoints

A quantized model adds one optional flat key, `quant`, holding a manifest
dict with `schema`, `recipe`, `keep_high_precision`, `execution`, calibration
provenance, `module_count` and `state`. FP8 manifests may also carry
`fp8_tensorwise_weights`, the exact list of `QuantLinear` module names whose
weight scale is tensorwise rather than per-output-channel. A loader that sees
`quant` rebuilds the quantized module structure and scaling policy before
`load_state_dict`.

`state` distinguishes the two artifact forms.

`"prepared"`, the default, holds FP32 master weights plus `_q_*` scale
buffers and is trainable. A reader without quantization support may ignore the
`quant` key and load the masters as a float model.

`"finalized"` is the deployment form written by `export(format="pt")`.
Masters are stripped and each quantized module instead carries packed weights:

| Recipe | Packed tensors | Dequantization |
|---|---|---|
| int8 | `weight_packed` int8 at the original weight shape, `_q_w_scale` FP32 per channel | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn at the original shape, `_q_w_scale` FP32 one entry per output channel | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, two 4-bit codes per byte, low nibble first, code `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, group 128 along in_features | Group-wise scale |
| int2 | Four 2-bit codes per byte, code `q + 2`, group 64 | Group-wise scale |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, code `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 per tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | As nvfp4 but 32-element blocks, plus `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Activation range buffers `_q_act_lo`, `_q_act_hi` and `_q_calibrated` are
retained for int8. The manifest records `remainder`, `"fp16"` or `"fp32"`,
for the non-quantized tensors. Unpacking reproduces the simulation bit for
bit, so finalized inference matches prepared inference exactly on the
finalizing device. This layout is the stable contract for external exporters
and runtimes.

## Training checkpoints

Trainer checkpoints use the same required metadata core and may add flat
training and resume fields:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` declares whether the top-level `model` is EMA-smoothed. When
EMA is enabled, `train_model`, `ema` and `ema_updates` preserve resume state.
Published inference weights should be lean and should not include optimizer,
epoch, config, loss or EMA resume state unless they are intentionally
distributed as training checkpoints.

For release compatibility, readers accept the legacy best-metric aliases
`best_mAP50_95`, `best_mAP50`, `best_metric` and `best_metric_name`.

## External snapshots

The schema governs LibreYOLO-authored `.pt` files. It does not rename or wrap
multi-file upstream snapshots used by the separate model tiers.

LibreMODUS size `14b-a7b` is an explicit exception: the alias resolves through
`LibreVLM(...)` to a directory of pinned upstream files, and LibreYOLO neither
adds v1.0 metadata to it nor republishes it as a `.pt`.

## Legacy and foreign weights

New writers validate strictly and must emit v1.0 metadata. When metadata is
missing or incomplete, legacy LibreYOLO-looking checkpoints load through the
compatibility path with a warning and conversion instructions, and foreign
upstream checkpoints route to auto-conversion. See
[upstream checkpoints](/docs/reference/upstream-checkpoints).

## Helpers

The schema helpers live in `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` is non-mutating and returns the list of
errors; with `strict=True` it raises `CheckpointMetadataError` instead.
`model.save(path)` is the supported way to write a conforming checkpoint.
