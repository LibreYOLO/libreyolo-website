---
title: CUDA graphs
seo_title: "LibreYOLO CUDA graph support matrix"
description: "Which families capture their forward at predict time and their forward and backward at train time, what the numbers are guaranteed to be, where a capture is split, and why an unsupported family raises."
lead: "A CUDA graph records one execution of a fixed sequence of kernels and replays it as a single launch. LibreYOLO captures inference on 39 verified families and training on 24, always per family, always after a bitwise parity check, and never as a silent fallback."
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - cuda graph support matrix
  - torch cuda graph training
  - capture_error_mode thread_local
  - cuda graph bit identical
last_verified: "1.5.0"
verification: "Inference family list derived from the CAPTURABLE matrix in tests/e2e/test_cuda_graph_families.py at v1.5.0. Training family list, parity classes and timings from docs/training_cuda_graphs.md. API and the NotImplementedError from BaseModel._require_cuda_graph_support, cuda_graph_scope and capture_graph in libreyolo/models/base/model.py, with the SUPPORTS_CUDA_GRAPH class variable. Seam splits read from the _get_graph_runner overrides in the depth_anything3, birefnet, ppocr, sam and sensenova families and from libreyolo/models/base/detr_cuda_graph.py. capture_error_mode from libreyolo/models/base/cuda_graph.py and libreyolo/training/cuda_graph.py. Training fallback from libreyolo/training/trainer.py and the --cuda-graph flag from libreyolo/cli/commands/train.py."
meta:
  - label: Inference families
    value: "39"
  - label: Training families
    value: "24"
  - label: Inference flag
    value: predict(cuda_graph=True)
    mono: true
  - label: Training flag
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Predict
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # True captures on first use per input shape.
        # "auto" waits until a shape repeats before paying the capture cost.
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Train
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Train from the CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
---

## What is captured

A graph records a fixed sequence of kernels and the memory addresses they read
and write. It does not record values, shapes or control flow. Replay is a single
launch instead of hundreds, which is why the gain is largest on small networks
at small batch sizes, where a step is dominated by launch overhead rather than
by arithmetic.

The two entry points capture different amounts of work.

| | Inside the graph | Eager |
|---|---|---|
| Inference | The network forward, `model._forward(x)` | Preprocessing, NMS, all postprocessing |
| Training | The network forward and backward | Loss, optimizer step, gradient clipping, EMA, LR schedule |

Neither NMS nor detection loss is a candidate. Both select with boolean masks,
run Hungarian matching or an assigner, and branch on the result, which is
exactly what a graph cannot record. Keeping them out is what makes capture safe
rather than a limitation to work around.

<code-tabs name="usage" />

`cuda_graph` accepts three values at predict time. `False` is the default.
`True` captures the first time each input shape is seen. `"auto"` waits for a
shape to repeat, so one-shot and shape-varying work never pays for a capture it
will not reuse. `capture_graph(imgsz=None, batch=1, dtype=None)` moves the cost
off the first request, `graph_info()` reports captured graphs and replay counts,
and `release_graphs()` frees them.

At train time the flag is a plain boolean, `--cuda-graph` on the CLI. See
[prediction performance](/docs/predict/performance) and
[training performance](/docs/train/performance) for the surrounding controls.

## Inference support

Support is per family, declared through the `SUPPORTS_CUDA_GRAPH` class
variable, and a family is only flagged after it captures and replays
bit-identically against two probe inputs drawn from different distributions.
That shared parity matrix covers 39 families across nine tasks.

| Task | Families |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Several families appear under more than one task, so the matrix runs more rows
than it has distinct families. Three more families capture through
family-specific code paths with their own dedicated tests rather than through
the shared matrix, and are not part of the 39: PP-OCR, SAM and SenseNova.

The verification is bitwise, not approximate. An earlier version of the protocol
judged parity by relative magnitude and wrongly demoted three healthy families,
YOLOX, EfficientNetV2 and YOLOv7, whose eager-to-graph difference measures
around 1e-7 while still being bit-identical on the probe that matters.

## Training support

Training capture went from two families to 24 in this release, across five
tasks.

| Task | Families |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Everything else trains eager: other tasks on those same families, families not
listed, distributed runs and distillation runs. Capture is also skipped while a
shape is still new, since the training path waits for an input shape to repeat
three times before capturing, which means `multi_scale=True` may never capture
at all.

## Two different answers for an unsupported family

The inference path raises. `predict(cuda_graph=True)` on a family that has not
opted in raises `NotImplementedError` naming the family, rather than running
eager and letting you believe you got a speedup you did not get. The reason is
that a bad capture does not fail loudly: replay of a forward that does something
uncapturable returns wrong numbers silently, so support has to be an explicit
per-family assertion rather than an attempt with a fallback.

The training path logs. `train(cuda_graph=True)` is always safe to pass, and a
family, task or configuration that cannot be captured writes one line and trains
eager, unchanged. A capture that fails partway through a run also drops the rest
of the run to eager rather than aborting it. The asymmetry is deliberate:
prediction is a call you can fix at the call site, while a training run should
not die at hour six over an optional optimization.

## Seam splitting

Some families cannot be captured whole because one stage genuinely does
something a graph cannot record. Rather than dropping the family, capture is
split at a verified seam: the capturable part replays, the rest runs eagerly,
and the combined output is the same as running everything eagerly.

| Family | Captured | Eager, and why |
|---|---|---|
| Depth Anything 3 | The network | The sky step, which is host-visible work after the forward |
| BiRefNet | The encoder, `forward_enc` | The decoder, whose `deform_conv2d` replays to a different result under capture |
| PP-OCR | The detection stage, `forward_det` | Recognition, because crop widths vary per line |
| SAM | The image encoder | The prompt path, which runs many times per encode |
| SenseNova | The vision tower | Autoregressive generation, with a KV cache that grows every step |
| Encoder-decoder detectors | Backbone and encoder | Decoder and Hungarian criterion |

The BiRefNet split is worth reading twice: `deform_conv2d` misbehaving under
capture reproduces on a bare call outside any model. Replacing it with a
pure-PyTorch equivalent was rejected because that would have shifted eager
predictions too, and eager numbers are the contract.

The encoder-decoder case covers D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 and EC. Their decoder builds contrastive-denoising queries from the
ground truth, and the number of those queries comes from the largest
ground-truth count in the batch, so the decoder's token count changes from batch
to batch. That is the one thing a graph cannot tolerate. Backbone plus encoder
is roughly a fifth to a quarter of a step for these families, which is why they
sit at the bottom of the speedup table.

PP-OCR captures one graph per detection input shape, bounded by the runner's
cache cap, and returns the eager result when no capture scope is active.

## Numerics

Most families are bit-identical, and where they are not, the reason is named
rather than waved at. At step zero of training the loss is bit-identical for all
24 families and no BatchNorm buffer differs; the gradient comparison is what
separates the categories.

| Class | Families | Meaning |
|---|---|---|
| Exact | Most of the 24 | Every gradient bit-identical |
| 1 ULP | fomo, lingbotvision | The last bit of float32, about 1e-7 relative, from a different summation order |
| Eager noise | The DETR lineage | Graphed differs from eager no more than two eager runs differ from each other |
| Float rounding | rtmdet | 137 of 139 gradients bit-identical, two differ by about 3e-4 |
| Own RNG stream | segformer | Stochastic depth sits inside the captured region |

The eager-noise class is the important one to read correctly. For those
families, two seeded eager runs already disagree, so bit-identical is not a bar
the graphed run failed; it is a bar nothing clears. That holds more widely at
`amp=False`, where a measured 3.2e-7 relative nondeterminism in an fp32 weight
gradient compounds: two seeded eager YOLOv9-t runs diverge by 36 percent over 20
steps, and turning TF32 off does not fix it.

## Pin memory

Capture runs with `capture_error_mode="thread_local"`. Under PyTorch's default
`"global"` mode, a DataLoader pin-memory thread staging the next batch calls
`cudaHostAlloc`, which both invalidates the in-flight capture and gets poisoned
by it, so the run dies on its next batch fetch with an error raised from inside
the pin-memory thread. That pairing was observed twice on a real training
campaign before it was diagnosed.

Thread-local mode restricts only the capturing thread. The pin thread never
touches the capturing stream, so nothing it does belongs in the graph in the
first place. Training goes further and temporarily substitutes a
`torch.cuda.CUDAGraph` subclass that forces the mode, because
`make_graphed_callables` exposes no argument for it, under a lock so two
concurrent captures cannot leave the substitution installed.

## What it is worth

Measured on an RTX 5070 Ti under AMP, one process per arm, replaying one real
batch so the dataloader is out of the loop, fastest of 24 steps after warm-up.
Detection at 640 px, classification at 224 px.

| Family | Batch | Speedup |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Everything else | varies | 1.04x to 1.26x |

A whole run gains less, because a graph cannot speed up the dataloader or
validation. A 20-epoch YOLOv9-t fine-tune on 406 images went from 428.4 s to
367.7 s, a 1.16x end-to-end gain, with an identical mAP50-95 of 0.6394 in both
arms and identical per-epoch losses.

The ceiling is set by how much of a step is network. On the same hardware at 640
px and batch 8, that is 84 percent for YOLOv9-t but only 26 percent for RTMDet-t,
which spends most of a step in its label assigner. Launch overhead is highest on
Windows, so Linux gains land at roughly a third to half of this table, and a
dataloader-bound run sees no wall-clock change at all. Peak memory moves between
5 percent lower and 19 percent higher.

## Caveats

A graph records addresses, not values, so anything that relocates parameters
drops it. Changing device through `predict(device=...)`, quantizing and
dequantizing all invalidate captured graphs.

Batch size matters more than family: RT-DETR-r18 gains 1.19x at batch 2 and
1.04x at batch 8, because a large batch is compute-bound and has less launch
overhead to remove.

The inference parity suite ran without the optional `kernels` package installed,
so capture safety with compiled Hub kernels active is not covered by it. Set
`LIBREYOLO_HUB_KERNELS=0` to take them out of the picture while isolating a
capture problem. See [kernels](/docs/reference/kernels).
