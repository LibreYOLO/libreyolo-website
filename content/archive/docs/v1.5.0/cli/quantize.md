---
title: libreyolo quantize
seo_title: "libreyolo quantize command reference"
description: "Quantize a checkpoint in PyTorch from the command line: recipes, calibration arguments, defaults, and the families each recipe accepts."
lead: "Replaces a model's float modules with quantized ones, calibrates them on unlabeled images where the recipe needs statistics, and saves the result as a PyTorch checkpoint."
keywords: [libreyolo quantize cli, int8 quantization command, fp8 quantization, post training quantization cli, libreyolo quantize arguments]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo quantize
    mono: true
  - label: Required
    value: model
    mono: true
  - label: Output
    value: "The source path with -<recipe> before the suffix, e.g. LibreYOLO9s-int8.pt"
    mono: true
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        # Calibrates on coco128 and writes LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: Cast only, no calibration
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: Wider calibration, then heal
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # Quantization-aware training on the quantized checkpoint recovers accuracy.
        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10 lr0=0.001
---

## Synopsis

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `recipe=int8` and
`--recipe int8` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `model` | | Model weights `.pt`. Required |
| `recipe` | `int8` | Quantization recipe: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Calibration images: a data YAML or a built-in dataset name. Unlabeled, forward only. `none` skips calibration |
| `samples` | `128` | Maximum calibration images |
| `batch` | `8` | Calibration batch size |
| `algorithm` | `auto` | Activation range estimation: `auto`, which selects minmax, or `minmax`, or `percentile` |
| `out` | | Output checkpoint path. Defaults to the source path with `-<recipe>` before the suffix |
| `device` | `auto` | Device |
| `allow_download_scripts` | `false` | Allow embedded Python in dataset YAML download blocks |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `help_json` | `false` | Dump command schema as JSON and exit |

## Examples

<code-tabs name="examples" />

## Notes

### Which families accept it

Quantization covers four families: `yolo9`, `rfdetr`, `birefnet` and
`feynobg`. Any other family exits with `quantize_failed` carrying the list.

### What each recipe touches

`fp16` and `bf16` are casts. They change dtype only, need no calibration, and
`calib=none` is the right setting for them.

`int8` and `fp8` quantize `Conv2d` and `Linear` modules, which is why they suit
the convolutional families.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` and `int2` quantize `nn.Linear` only, so they
target the transformer families. Asking for one of them on `yolo9` is refused
with an explanation rather than silently producing an unquantized model, since
sub-8-bit acceleration there is GEMM only and the convolutions would stay in
higher precision.

`int8`, `fp8`, `w4a8` and `int2` need calibration statistics for their
activations. `int2` also needs training to heal afterwards, so it is refused on
`birefnet` and `feynobg`, which have no trainer.

Each family keeps a set of modules in float regardless of recipe: first layers,
prediction heads, and on YOLOv9 the DFL convolution, which is a fixed integral
expectation operator that must not be quantized.

### Calibration data is not training data

`calib` points at a small unlabeled image set, used forward only, to derive
activation ranges. It is not evaluated against and its labels are never read.
The default `coco128.yaml` downloads on first use from a URL, so it needs no
extra permission; a YAML with an embedded Python download script needs
`allow_download_scripts=true`.

`algorithm=percentile` is available and can reduce accuracy on transformer
families, which is why `auto` selects minmax.

### Recovering accuracy

The output is a normal PyTorch checkpoint, so
[`libreyolo train`](/docs/cli/train) accepts it directly. Training a quantized
checkpoint is quantization-aware training; adding `distill_model=<teacher>`
makes it quantization-aware distillation.

### Output and exit codes

The result prints the saved path, the recipe, the execution mode, whether
calibration ran, and the count of modules swapped per kind. The exit code is
`0` on success, `4` when the model cannot be loaded, `5` when quantization or
the save fails, and `1` for other runtime failures.

Related: [`libreyolo export`](/docs/cli/export), which leaves PyTorch and writes
a deployment artifact instead.
