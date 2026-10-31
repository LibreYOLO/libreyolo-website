---
title: libreyolo profile
seo_title: "libreyolo profile command reference"
description: "Measure training and inference speed and read the result: every profile subcommand, its arguments and defaults, and what each lens reports."
lead: "A command group that measures where time goes in a training step or an inference call, writes a self-contained profile, and reads that profile back through several lenses."
keywords: [libreyolo profile cli, training profiler, inference latency profiling, gpu kernel profiling, libreyolo profile compare]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo profile
    mono: true
  - label: Output
    value: "profile.json and profile_trace.json under runs/profile"
    mono: true
snippets:
  examples:
    - label: Measure inference
      language: bash
      code: |
        # No source argument means the bundled sample image.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Read the verdict
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Compare two measurements
      language: bash
      code: |
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project runs/profile/a
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4 --project runs/profile/b

        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
---

## Synopsis

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

This group does not take `key=value` arguments. Its subcommands use positional
arguments and POSIX flags, so it is `--weights LibreYOLO9t.pt`, not
`weights=LibreYOLO9t.pt`. Running `libreyolo profile` with no subcommand prints
the list.

Two subcommands measure and write a profile; the rest read one. `run` and
`infer` both emit the same self-contained `profile.json`, so every reading
subcommand works on either.

## profile run

Runs a short profiled training and writes a profile.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argument | Default | Meaning |
|---|---|---|
| `data` | | Positional. Dataset YAML or name, e.g. `coco128`. Required |
| `--weights` | `LibreYOLO9t.pt` | Model weights file or name |
| `--size` | `t` | Model size variant |
| `--batch` | `16` | Micro-batch. `-1` auto-fits about 70% of VRAM |
| `--imgsz` | `640` | Training image size |
| `--workers` | `8` | Dataloader workers |
| `--amp` | `true` | Use the family's AMP path. `--no-amp` disables it |
| `--steps` | `20` | Profiled, that is measured, steps |
| `--warmup` | `5` | Warmup steps before measuring |
| `--repeat` | `1` | Repeat N times for a mean and standard deviation |
| `--device` | `0` | Device |
| `--project` | `runs/profile` | Output directory root |
| `--json` | `false` | JSON output to stdout |

The measured window is `--warmup` plus `--steps` iterations. A dataset too
small to fill it produces no profile and the command exits with code `3`,
naming the three ways out: a larger dataset, fewer steps, or a smaller batch.

`--repeat` above 1 writes an aggregated `runs/profile/profile_repeat.json`
whose scalar metrics are averaged across trials, while the kernel lists come
from the final trial. It is also the prerequisite for a significance verdict in
`compare`: a single run cannot supply one.

## profile infer

Profiles the inference path and writes a profile.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | | Positional. Image or directory. The bundled sample image when omitted |
| `--weights` | `LibreYOLO9t.pt` | Model weights file or name |
| `--size` | `t` | Model size variant |
| `--batch` | `1` | Images per forward pass |
| `--imgsz` | `640` | Input image size |
| `--half` | `false` | Autocast forward, CUDA only. `--no-half` disables it |
| `--amp-dtype` | `float16` | CUDA autocast dtype: `float16` or `bfloat16` |
| `--warmup` | `20` | Warmup iterations before measuring |
| `--runs` | `100` | Measured iterations |
| `--repeat` | `1` | Repeat N times for a mean and standard deviation |
| `--conf` | `0.25` | Confidence threshold, which changes how much work NMS does |
| `--iou` | `0.45` | NMS IoU threshold |
| `--max-det` | `300` | Max detections per image, which changes how much work NMS does |
| `--device` | `0` | Device |
| `--trace` | `true` | Emit a Chrome trace for kernel and op drill-down. `--no-trace` skips it |
| `--project` | `runs/profile` | Output directory root |
| `--json` | `false` | JSON output to stdout |

Reports latency at p50, p90 and p99, throughput in images per second, and the
stage split across preprocess, forward and postprocess. The three threshold
arguments are here because they move the postprocess number.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argument | Default | Meaning |
|---|---|---|
| `trace` | | Positional. Path to a `profile.json` or `profile_trace.json`. Required |
| `--json` | `false` | JSON output to stdout |

The high-level read: step time, throughput, GPU utilization, Tensor Core share,
peak VRAM, host overhead, kernel launches per step, the bottleneck verdict with
its reason, the kernel mix by category, and the top kernels per step. On an
inference profile it also prints the latency percentiles and the stage split.

A profile taken under VRAM thrash is marked, because utilization and throughput
measured there cannot be trusted.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argument | Default | Meaning |
|---|---|---|
| `trace` | | Positional. Path to a profile. Required |
| `field` | | Positional. Metric name. Omit to list the available metrics |
| `--json` | `false` | JSON output to stdout |

Prints one metric and nothing else, for scripted loops. An unknown field exits
with code `2` and points at the listing form.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argument | Default | Meaning |
|---|---|---|
| `trace` | | Positional. Path to a profile. Required |
| `--json` | `false` | JSON output to stdout |

GPU milliseconds, wall milliseconds, kernel count and op count per phase:
forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argument | Default | Meaning |
|---|---|---|
| `trace` | | Positional. Path to a profile. Required |
| `--top` | `20` | Show top N by GPU time |
| `--category` | | Filter by category substring: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Filter by kernel-name regular expression |
| `--tensorcore` | `false` | Only Tensor Core kernels |
| `--sort` | `time` | `time`, `count` or `name` |
| `--phase` | | Restrict to one phase: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | JSON output to stdout |

The bottom of the analysis: individual GPU kernels with their share of GPU
time, milliseconds per step, invocations per step and category. An unknown
`--phase` exits with code `2` and lists the phases the profile has.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argument | Default | Meaning |
|---|---|---|
| `trace` | | Positional. Path to a profile. Required |
| `--top` | `20` | Show top N by CPU time |
| `--phase` | | Restrict to one phase |
| `--json` | `false` | JSON output to stdout |

The framework view rather than the device view: `aten` and autograd ops ranked
by CPU time, which is where host-launch cost shows up.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argument | Default | Meaning |
|---|---|---|
| `before` | | Positional. Baseline profile. Required |
| `after` | | Positional. New profile. Required |
| `--json` | `false` | JSON output to stdout |

Diffs throughput, milliseconds per image, GPU utilization, host overhead,
kernel launches per step and the bottleneck verdict.

The significance call needs both sides measured with `--repeat` of at least 2.
Given that, a difference counts as significant when it exceeds twice the
combined standard error, and the output prints the comparison it made. Without
it, the line reads that a single run cannot support the call.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argument | Default | Meaning |
|---|---|---|
| `trace` | | Positional. Path to a profile. Required |
| `--remove-category` | | Project removing a kernel category: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Project removing N kernel launches per step, for example an op-fusion win |
| `--json` | `false` | JSON output to stdout |

Estimates what a change would buy before the change is written. One of the two
options is required; neither exits with code `2`.

The projection follows the profile's own verdict. Below 80% GPU utilization it
models the saving as fewer launches times the measured per-launch host cost;
above it, as less GPU work. The result carries a caveat field, because the
per-launch cost is an approximation and the only proof is a second measurement.

## Examples

<code-tabs name="examples" />

## Notes

The profiler measures and reports. It changes nothing: reading the verdict,
editing the configuration or the code, re-running, and comparing is the loop it
is built for.

`--device` defaults to `0`, which is CUDA device 0. Passing `--device cpu`
measures on the CPU and produces a profile the reading subcommands still
accept, without the GPU kernel detail.

Every subcommand supports `--json`, and the reading ones print to stdout only,
which is what makes the group usable from a script.

Exit codes here are the group's own: `2` for a file that does not exist or an
argument that does not resolve, `3` when `run` produced no profile, and `1`
when a trace cannot be analyzed.

Related: [`libreyolo train`](/docs/cli/train), whose arguments are what a
training profile is usually taken to tune.
