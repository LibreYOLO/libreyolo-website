---
title: libreyolo doctor
seo_title: "libreyolo doctor command reference"
description: "Check a detection dataset before training: arguments with defaults, the check families you can skip or select, and the exit codes CI can gate on."
lead: "Runs a set of health checks over a detection dataset and reports what would hurt a training run: missing files, broken labels, corrupt images, split leakage and class imbalance."
keywords: [libreyolo doctor cli, dataset health check, yolo dataset validation, dataset leakage check, libreyolo doctor strict]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo doctor
    mono: true
  - label: Required
    value: data
    mono: true
  - label: Output
    value: "A findings report on stdout. Exit 1 when errors are found"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        # download=true lets the bundled coco8.yaml fetch its images if missing.
        libreyolo doctor coco8.yaml download=true
    - label: Fast pass, no image decoding
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: CI gate on selected checks
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
---

## Synopsis

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

The dataset is positional, and `data=<path>` is accepted as an alternative.
Giving both with different values exits with `config_conflict`. Everything else
is a `key=value` pair, and POSIX form works too, so `imgsz=1024` and
`--imgsz 1024` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `data` | | Positional. Dataset YAML in YOLO detection format, e.g. `coco8.yaml`. Required |
| `imgsz` | `640` | Training image size used for pixel-based checks such as tiny objects |
| `fast` | `false` | Skip image decoding, which drops the corruption, duplicate and leakage checks |
| `skip` | | Comma-separated check ids or families to skip, e.g. `images,labels.tiny_object` |
| `only` | | Comma-separated check ids or families to run exclusively |
| `strict` | `false` | Warnings also fail the exit code, for CI gates |
| `download` | `false` | Allow URL-based dataset download if missing. Never scripts |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `help_json` | `false` | Dump command schema as JSON and exit |

### Check families

`skip` and `only` accept either a full check id or a family prefix, so `images`
selects every `images.*` check.

| Family | Covers |
|---|---|
| `config` | The dataset YAML itself: missing `names`, `nc` against `names`, missing splits, unresolvable `path`, duplicate class names |
| `files` | Image and label pairing: missing labels, missing images, orphan labels, unsupported extensions, case collisions |
| `labels` | Label content: syntax, polygon lines, class ids out of range, coordinates out of range, degenerate boxes, tiny objects, huge boxes, extreme aspect ratios, duplicate boxes, crowded images, identical files |
| `images` | Pixel data: corrupt files, EXIF orientation, unusual color modes, tiny or extreme dimensions, uniform images, exact and near duplicates |
| `splits` | Leakage between splits, exact and near |
| `balance` | Class distribution: classes with zero or few instances, imbalance, split coverage, background ratio, split skew |

## Examples

<code-tabs name="examples" />

## Notes

### Exit codes

`0` when no errors were found, `1` when any finding is an error. With
`strict=true`, warnings raise the exit code to `1` as well, which is the
setting a CI gate wants.

Usage problems have their own codes: `2` for an unknown check id or family in
`skip` or `only`, `3` when the dataset cannot be found, and `3` when the
dataset is not detection shaped.

### Selection resolves before the scan

`skip` and `only` are resolved against the check registry before anything is
read from disk, so a typo fails immediately rather than after a long image
pass. A selector that matches nothing is an error, and the message lists the
known families.

If the combination of `skip`, `only` and `fast` leaves no checks to run, that
is also an error rather than a silent pass.

### Downloads

The dataset is not fetched unless `download=true`, and only URL downloads are
ever performed. An embedded Python download script in a dataset YAML is never
executed by this command, whatever the flag.

### Scope

The checks are written for detection datasets. A dataset whose labels are pose,
segmentation or oriented-box shaped is detected and refused with `data_invalid`
rather than scored against the wrong rules.

### Output

The human report goes to stdout, and `json=true` replaces it with a structured
object carrying the summary counts, the dataset statistics, every finding, and
the list of checks that were skipped.

Related: [`libreyolo train`](/docs/cli/train), the run this command is meant to
be run before.
