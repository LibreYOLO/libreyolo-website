---
title: Training performance
seo_title: "Faster training: CUDA graphs, AMP, profiler"
description: "Make a training run faster: capture the step into CUDA graphs, pick an AMP dtype, and use the built-in profiler to find where the time actually goes."
lead: "Three levers change how fast a training step runs: mixed precision, CUDA graph capture of the network's forward and backward, and whatever the profiler says is actually holding the step up."
keywords:
  - cuda graphs training
  - training speed
  - mixed precision training
  - bfloat16 training
  - pytorch profiler
  - dataloader bound
  - kernel launch overhead
  - gpu utilization
last_verified: "1.5.0"
snippets:
  profile:
    - label: Profile and keep training
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Profiles a short window of real steps, prints a verdict, then
        # continues the run with the hooks removed.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: Measure only, then stop
      language: bash
      code: |
        # Sets no_aug_epochs=0 and runs just enough epochs to fill the window.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Drill into the result
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
---

## Measure before changing anything

The three levers below fix different problems, and applying the wrong one
changes nothing. The profiler says which problem you have.

<code-tabs name="profile" />

`profile=True` measures a window of real training steps, five discarded then
twenty measured by default, prints a report, writes its artifacts and then keeps
training with the hooks dropped. It costs nothing when off, and it is ignored
under distributed training.

The report ends in one of four verdicts:

| Verdict | Meaning | Levers |
|---|---|---|
| `dataloader` | the GPU waits on input data | more `workers`, `cache="ram"` or `"disk"`, lighter augmentation, larger batch |
| `host / launch` | the GPU is fed too slowly, many tiny kernels | larger batch, CUDA graphs, fewer per-step host syncs |
| `compute` | the GPU is saturated | AMP or bfloat16, or accept it |
| `memory-pressure` | allocator thrash, VRAM at the edge | lower batch; utilization figures here are unreliable |

The utilization number is kernel busy time over the unsynchronized step time.
The window is deliberately split: the first half runs with no extra
synchronization so the verdict reflects real overlap, and only the second half
brackets each phase with a sync to attribute GPU time. Synchronizing every phase
hands the dataloader workers slack and hides starvation, so the composition
numbers are never used to pick the verdict.

Four files land in the run directory: `timeline.html`, which opens in a browser
by itself, `profile_trace.json` for Perfetto or Nsight, `profile_summary.json`,
and `profile.json`, the self-contained one to copy around and feed back to the
`libreyolo profile` subcommands.

Two things about `profile run` are worth knowing. It sets `no_aug_epochs=0`,
because the profiler measures epoch 0 and a short run with the default
`no_aug_epochs` would profile the lighter no-augmentation dataloader rather than
the one training actually uses. And `--repeat N` reports mean and standard
deviation, which matters because a launch-bound step is noisy enough that a
single run misleads; it writes per-trial directories `prof_1`, `prof_2` and so
on, plus an aggregate `profile_repeat.json`.

## Mixed precision

`amp=True` is the default for most families and runs the forward pass under CUDA
autocast. `amp_dtype` chooses `float16` or `bfloat16`.

<code-tabs name="amp" />

Float16 needs dynamic loss scaling and gets a live gradient scaler; bfloat16's
wider exponent range does not, so its scaler is disabled. Four families ship with
`amp=False`, D-FINE, DEIM, YOLO-NAS and FOMO, and the DEIM setting carries
through to RT-DETRv4 by inheritance. D-FINE states the reason: its decoder clamps
activations at 65504, the largest finite float16 value.

The argument semantics, including what a bfloat16 request does on hardware
without bfloat16 support, are on
[Hyperparameters](/docs/train/hyperparameters).

## CUDA graphs

`cuda_graph=True` captures the network's training forward and backward into a
CUDA graph, removing per-step kernel launch overhead.

<code-tabs name="graph" />

The flag is always safe to pass. A family, task or configuration that cannot be
captured logs one line and trains eager, unchanged.

Only the network is captured. The loss stays eager by design, because detection
losses select with boolean masks, run Hungarian matching and branch on assignment
results, none of which a graph can record. The optimizer step, gradient clipping,
EMA update and learning-rate schedule stay eager too.

That bounds the win by how much of a step is network, and the share varies
widely. Measured on an RTX 5070 Ti at 640 px, batch 8: 84 percent of a YOLOv9-t
step is network, 44 percent of a YOLOv7-b step, 31 percent of a YOLOX-t step and
26 percent of an RTMDet-t step. The last two spend most of a step inside their
label assigners, so capturing the network helps them least.

### What it is worth

Conditions for every figure below: RTX 5070 Ti, Windows, AMP, one process per
arm from a shared saved state, replaying one real batch so the dataloader is out
of the loop, fastest of 24 steps after warm-up. Detection at 640 px,
classification at 224 px. Batch size is per row.

| Family | Size | Batch | Eager | Graphed | Speedup |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Those isolate the GPU step. A complete fine-tune also pays for the dataloader and
for validation. YOLOv9-t on a 406-image detection set, 20 epochs, batch 8, 640 px,
4 dataloader workers, on the same machine: 428.4 s wall clock eager against
367.7 s graphed, a 1.16x gain, with mAP50-95 of 0.6394 in both arms.

Three things move these numbers. Small batches are launch-bound and large ones
are compute-bound, so RT-DETR-r18 gains 1.19x at batch 2 and 1.04x at batch 8.
Launch overhead is highest on Windows, and Linux gains are roughly a third to
half of the table. And a dataloader-bound run sees no wall-clock change at all,
which is why the profiler comes first.

Capture engages the same way at `amp=False`, but fp32 kernels run longer, so a
step is less launch-bound and most families gain less. On the same hardware,
MobileNetV4-s at batch 16 goes from 2.74x under AMP to 3.61x at fp32, while
YOLOv9-t at batch 8 goes from 1.99x to 1.69x and RT-DETR-r18 at batch 4 from
1.12x to 0.99x.

### Where capture applies

| Task | Families |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Everything else falls back to eager with one log line: other tasks on those
families, families not listed, distributed runs and distillation runs. A capture
failure at runtime also drops the rest of the run to eager rather than failing.

For the encoder-decoder detectors, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 and v4,
and EC, only the backbone and encoder are captured. Their decoder reads the
ground truth to build contrastive-denoising queries, and the number of those
queries follows the largest ground-truth count in the batch, so its token count
changes from batch to batch.

### Shapes

A graph is valid for exactly the input shape it was captured with. The trainer
counts batch shapes and captures once a shape has repeated three times. Batches
at any other shape run eager: multi-scale batches, and the last partial batch of
an epoch.

This is the trap for the DETR families, which resize every batch by default. With
`multi_scale=True` a short run may never see one shape often enough to capture at
all. Pass `multi_scale=False` when the speedup is the point.

YOLOX changes what the captured region computes partway through a run, turning on
its L1 regression branch when mosaic closes at `no_aug_epochs`. The trainer
invalidates the capture there and re-captures once the new shape settles.

### Numerics and memory

Most families reproduce their eager loss trajectory bit for bit under AMP. FOMO
and LingBot-Vision differ in the last bit of float32 from a different summation
order. The deformable-attention detectors, D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR
and EC, do not reproduce their own eager runs either, because that backward
accumulates with atomics and TF32 convolutions pick a reduction order per launch;
the graphed run stays inside that spread. RTMDet differs by roughly 3e-4 relative
on two of 139 gradients, because it shares head convolutions across pyramid
levels and the two backward paths sum three contributions in a different order.
SegFormer has stochastic depth inside the captured region, so a replayed graph
draws its own random stream and is statistically equivalent to eager rather than
identical; the manager logs that once at capture time.

At `amp=False` bit-identical is not available from anything on this hardware,
with or without capture. Two identical seeded eager YOLOv9-t runs diverge by
36 percent relative over 20 steps and YOLOX-t by 2.6 percent, because cuDNN
picks a nondeterministic weight-gradient algorithm for some fp32 convolution
shapes.

A captured graph pins static input, output and workspace buffers, so peak VRAM
rises by roughly one extra set of activations. Across the families above, peak
allocation moved between -5 and +19 percent. The relative cost is largest for the
small classification models, whose activations are small to begin with: ResNet-18
at 224 px, batch 16, went from 0.48 GB eager to 0.57 GB graphed. If it pushes a
run over the limit, lower the batch or leave the flag off.

## Related

- [Hyperparameters](/docs/train/hyperparameters) for `batch`, `nbs`, `cache` and
  `workers`.
- [Multi-GPU training](/docs/train/multi-gpu), where both CUDA graphs and the
  profiler are unavailable.
- [CUDA graphs](/docs/reference/cuda-graphs) for the combined inference and
  training support matrix, the seam splits and the numerics contract.
