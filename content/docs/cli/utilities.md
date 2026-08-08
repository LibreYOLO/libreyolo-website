---
title: libreyolo utilities
seo_title: "libreyolo CLI utility commands reference"
description: "The small LibreYOLO commands: version, checks, models, formats, cfg, info, metadata, enroll and compare, each with its arguments and defaults."
lead: "Nine commands that report or inspect rather than compute. They print environment facts, the model and format inventory, resolved defaults, checkpoint details, and they build and query a face gallery."
keywords: [libreyolo version, libreyolo checks, libreyolo models, libreyolo formats, libreyolo cfg, libreyolo info, libreyolo metadata, libreyolo enroll, libreyolo compare]
last_verified: "1.5.0"
meta:
  - label: Commands
    value: version, checks, models, formats, cfg, info, metadata, enroll, compare
    mono: true
  - label: Output
    value: "stdout, in text or with json=true as one object carrying schema_version"
snippets:
  examples:
    - label: Environment
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: What is available
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Inspect a checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
---

## Synopsis

```bash
libreyolo <command> [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `model=x` and
`--model x` are the same argument. Every command here writes results to stdout
and accepts `json=true` and `quiet=true`.

The root command carries one flag of its own, `libreyolo --version`, which
prints the version string and exits. That is a smaller output than the `version`
command below.

## version

Prints the LibreYOLO version plus the Python, torch and CUDA versions it is
running against.

```bash
libreyolo version
```

| Argument | Default | Meaning |
|---|---|---|
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

## checks

Prints the environment in more detail: Python, torch, CUDA, cuDNN, every
detected GPU with its name and memory, and the installed version of each
optional package the export paths use.

```bash
libreyolo checks
```

| Argument | Default | Meaning |
|---|---|---|
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

The package list covers `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` and `scipy`. A package that is not installed reports as such
rather than being omitted, so a failed export can be traced to a missing
dependency from this one command.

## models

Lists every model family with its tasks, sizes, the CLI names that resolve to
its checkpoints, and each size's input resolution.

```bash
libreyolo models
```

| Argument | Default | Meaning |
|---|---|---|
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

A family whose optional dependency is not installed is listed as unavailable
together with the `pip install` line that would make it available. The CLI
names are what `model=` accepts as a shorthand: `yolox-s` resolves to
`LibreYOLOXs.pt`, and non-detection tasks carry their task suffix.

## formats

Lists the export formats the installed environment can produce, with each
format's file extension and whether it supports FP16 and INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argument | Default | Meaning |
|---|---|---|
| `family` | | Show tiers for one model family. `model=` is accepted as the same option |
| `task` | | Canonical model task. The family's default task when unset |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

Without `family`, the output is the format inventory alone. With it, each
format gains the support tier for that family and task, the reason behind the
tier, and any constraint attached to it. An unknown family, or a task the family
does not support, is a usage error.

Format aliases appear next to their canonical name: `engine` for `tensorrt`,
`litert` for `tflite`.

## cfg

Prints the resolved default configuration: the train defaults, the validation
defaults, the prediction defaults, and the per-family overrides.

```bash
libreyolo cfg
```

| Argument | Default | Meaning |
|---|---|---|
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

The values are read from the configuration dataclasses, not from a copy, so
this is the authority on what a training run will use when you do not pass an
argument. `family_overrides` is the section that answers why a family trained
at settings you did not ask for. See
[`libreyolo train`](/docs/cli/train) for how those overrides are applied.

## info

Loads a model on the CPU and reports its family, size, parameter count,
classes, and the export tier for each format.

```bash
libreyolo info model=<name|path>
```

| Argument | Default | Meaning |
|---|---|---|
| `model` | | Model name or path to weights. Required |
| `detailed` | `false` | Include per-parameter details |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

## metadata

Reads a checkpoint's metadata without constructing a model, and validates it
against the LibreYOLO checkpoint schema.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argument | Default | Meaning |
|---|---|---|
| `path` | | Path to a `.pt` checkpoint. Required |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

Large tensor-bearing entries are summarized rather than printed, so the output
stays readable on a full training checkpoint. A checkpoint that does not exist
exits with `checkpoint_not_found`, and one whose metadata fails validation
prints the errors and exits `1`.

## enroll

Builds a face gallery from a folder-per-person tree, so later predictions can
name the faces they find.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argument | Default | Meaning |
|---|---|---|
| `model` | | Face-embedding model, path or name. Required |
| `source` | | Folder-per-person tree, `source/<identity>/*.jpg`. Required |
| `gallery` | | Output gallery file `.npz`. Extended in place if it exists. Required |
| `face_detector` | | Face detector: a YuNet `.onnx` or a LibreYOLO detector. The family's default detector when unset |
| `device` | `auto` | Device: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

```bash
# people/ holds one folder per identity; the folder name becomes the identity.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

The subfolder name is the identity. A reference image with no detectable face
is skipped with a line on stderr and the rest continue; a source with no
identity subfolders, or one where no face was found at all, is an error.

Pass the resulting file to
[`libreyolo predict`](/docs/cli/predict) as `gallery=people.npz` to have
detections carry an identity and a match score.

## compare

Reports the cosine similarity between two face images and whether it clears the
same-identity threshold.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argument | Default | Meaning |
|---|---|---|
| `model` | | Face-embedding model, path or name. Required |
| `source` | | First image. Required |
| `source2` | | Second image to compare against. Required |
| `face_detector` | | Face detector: a YuNet `.onnx` or a LibreYOLO detector |
| `threshold` | `0.4` | Cosine-similarity threshold for the same-identity decision |
| `device` | `auto` | Device: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` is registered as a second name for this command and takes
the same arguments.

Both `compare` and `enroll` need a model whose task is face embedding. Anything
else exits with `config_unsupported`. Local image paths and `http` or `https`
URLs are both accepted as sources.

## Examples

<code-tabs name="examples" />

## Notes

stdout carries the result; progress and warnings go to stderr. `json=true`
prints one object with `schema_version`, which is the form to read from a
script. Text output is the default and is meant to be read by a person.

Exit codes follow the same map as the rest of the CLI: `0` on success, `2` for
a usage or configuration error, `3` when a source cannot be found, `4` when a
model or checkpoint cannot be loaded, and `1` for other runtime failures.

Related: [`libreyolo doctor`](/docs/cli/doctor), which is the dataset-side
inspection command, and [`libreyolo profile`](/docs/cli/profile), the
performance-side one.
