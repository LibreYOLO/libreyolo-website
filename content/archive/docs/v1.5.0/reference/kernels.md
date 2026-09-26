---
title: Kernels
seo_title: "LibreYOLO kernel registry and Hub kernels"
description: "How LibreYOLO selects accelerated implementations: the kernel registry under libreyolo/kernels, the optional Hugging Face Hub MS-deform-attn kernel, and the fused attention switch."
lead: "Every accelerated operation in LibreYOLO has a portable default and, sometimes, a faster variant registered on top of it. Selection happens at runtime by predicate, a missing optional dependency is a fallback rather than an error, and an exported graph always takes the portable path."
keywords:
  - libreyolo kernels
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - hub-kernels extra
  - ms_deform_attn kernel
  - set_fused_attention
  - libreyolo triton kernels
last_verified: "1.5.0"
verification: "Registry API read from libreyolo/kernels/__init__.py at v1.5.0, attention API from libreyolo/kernels/attention/__init__.py and sdpa.py, Hub provider from libreyolo/kernels/attention/ms_deform_attn.py including its pinned revision and eligibility predicate. Directory layout listed from libreyolo/kernels/. Extra definition from pyproject.toml. Behavior notes and benchmark figures from docs/kernels.md. The v1.4.0 gating history from the RF-DETR slot-wiring commit and the 1.5.0 CHANGELOG entry."
meta:
  - label: Package
    value: libreyolo.kernels
    mono: true
  - label: Opt-in extra
    value: libreyolo[hub-kernels]
    mono: true
  - label: Force reference
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: See what is selected
      language: python
      code: |
        import libreyolo.kernels as kernels

        # Op slot to selected implementation name, or "unavailable".
        print(kernels.active())
    - label: Force the reference path
      language: bash
      code: |
        # off and reference both mean the same thing, and also skip
        # importing the accelerated providers at all.
        LIBREYOLO_KERNELS=off python train.py
    - label: Turn off Hub kernels without uninstalling
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Switch a family to fused attention
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Returns how many attention modules switched.
        print(set_fused_attention(model))
    - label: Register your own
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
---

## The registry

`libreyolo/kernels/` is a small runtime registry of pluggable implementations.
An op slot is a name such as `fake_quant_fp8` or `ms_deform_attn`. Callers ask
the registry for a slot and get back whichever registered implementation passes
its predicate first, newest registration winning, falling through to the
reference implementation when nothing else applies.

That structure exists so that an optional dependency is never a hard
requirement. A machine without Triton, without CUDA, or without the `kernels`
package runs the same code and produces the same numbers, only slower.

| Function | Purpose |
|---|---|
| `active()` | Op slot to selected implementation name, or `"unavailable"` |
| `resolve(op)` | The callable that would run, or `None` |
| `register(op, impl, *, name, predicate=None)` | Add an implementation, newest first |
| `unregister(op, name)` | Remove one |
| `clear_cache()` | Drop the memoized resolution |

<code-tabs name="usage" />

A predicate that raises is caught and warned about, never propagated, so a
broken third-party implementation degrades to the portable path instead of
breaking prediction.

### Layout

The tree is organized by purpose first and backend second, so a slot is found by
what it computes rather than by which library happens to implement it today.

| Directory | Contents |
|---|---|
| `kernels/quant/simulate/` | Fake-quantization Triton kernels, with straight-through backward, on any device. Used by QAT and by simulated post-training quantization alike |
| `kernels/quant/execute/` | Real-precision paths for finalized models only, no backward: the FP8 tensor-core GEMM, its fused Triton prologue and epilogue, and the packed-weight unpack kernels |
| `kernels/attention/` | Attention ops shared across families: the `ms_deform_attn` slot, and the fused-SDPA policy |

The boundary between `simulate` and `execute` is whether the model is finalized,
not whether it is training or deploying. The reference implementations stay in
`libreyolo/quant/`, which defines what the numbers mean; `kernels/` only makes
them fast. Weight packing has no variants at all, because it is the checkpoint
contract.

GEMM and attention slots have no reference implementation. A caller has to check
that `resolve()` returned something and keep its own portable path, which is why
ONNX, TensorRT and `torch.export` graphs always contain the portable math.

### Selection overrides

`LIBREYOLO_KERNELS=off` or `=reference` forces reference implementations and
short-circuits the import of the accelerated providers entirely. Any other value
restricts selection to implementations registered under that name.
`LIBREYOLO_QUANT_KERNELS` is honored as a legacy alias from when the registry
lived under `libreyolo/quant/`, and is read only when `LIBREYOLO_KERNELS` is
unset. Both are listed with the rest on [settings](/docs/reference/settings).

## Hub kernels

Compiled CUDA kernels published on the Hugging Face Hub load at runtime through
the optional `kernels` package. Nothing is vendored into LibreYOLO; the artifact
is fetched and cached by that package, and each provider pins an audited commit
revision, so bumping a pin requires a GPU parity run before it lands.

Installing the extra is the opt-in:

```bash
pip install "libreyolo[hub-kernels]"
```

Without the package nothing changes and no network request is made.
`LIBREYOLO_HUB_KERNELS=0` disables the fetch without uninstalling anything. A
kernel that fails to load or to run disables itself for the rest of the process
and falls back with one warning.

One slot is Hub-backed today: `ms_deform_attn`, the compiled multi-scale
deformable attention forward and backward from Deformable DETR, under Apache
2.0. It is wired into the whole deformable lineage: RF-DETR, Deformable DETR,
DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM,
DEIMv2, EC and OV-DEIM. Because the backward is compiled too, training benefits
as well as prediction.

Eligibility is narrow on purpose. Inputs must be CUDA and float32, and execution
must be eager: the provider declines under `torch.jit.is_tracing()`,
`torch.compiler.is_compiling()`, `torch.compiler.is_exporting()` and
`torch.onnx.is_in_onnx_export()`. Two input layouts also fall through to the
portable path, a per-level point count that varies between levels, and discrete
integer-index sampling. The EC pose variant is not wired.

### This kernel is newly reachable

Read this before installing the extra on an existing project.

In v1.4.0 the slot was consulted from inside a helper, behind a condition that
required the spatial-shape pairs to be absent. RF-DETR always threads those
pairs through its decoder, so the condition never held and the kernel never
executed in any eager forward. The consult moved in v1.5.0, and the kernel now
actually runs.

The practical consequence is that upgrading to v1.5.0 *and* installing
`libreyolo[hub-kernels]` on CUDA means RF-DETR and its lineage take their
forward from a compiled binary for the first time. Predictions and metrics can
shift at float tolerance as a result. A stock install, without the extra, is
unaffected. If you are comparing metrics across the upgrade, hold the extra
fixed or set `LIBREYOLO_HUB_KERNELS=0` on both sides.

## Fused attention

Fused scaled dot-product attention needs no optional dependency, only stock
PyTorch, so it is governed by policy rather than by availability. Two rules
apply.

First, a graph capture never uses it. Every swapped call site keeps the
primitive-op equation available behind an export check, covering ONNX export,
whose default opset has no SDPA symbolic, and `torch.jit.trace`, which
TorchScript, CoreML and NCNN all go through. Dynamo captures are deliberately
outside the gate, because `torch.compile` lowers SDPA better than the manual
math, and both Core AI and ExecuTorch decompose SDPA to core ATen on their own.

Second, the parity bar for making it the default is byte exact. Families that
clear it use SDPA by default: SegFormer, Depth Anything and MoGe-2, BERT,
Grounding DINO, SwinIR and PP-OCR. Families that do not keep manual math and
expose a `fused_attn` flag instead, which is what `set_fused_attention(model)`
flips: Swin, DINO-DETR's Swin backbone, BiRefNet and FeyNobg, OWLv2, LW-DETR,
SigLIP 2, ZipDepth and MobileSAM. ViT and DeiT carry the same flag but default
it on, following upstream, so the same call with `enabled=False` turns them off.

It is worth doing where it applies. On an RTX 5070 Ti under fp16 autocast, Swin
window attention goes from 1.278 ms to 0.721 ms, a 1.77x gain, and OWLv2 vision
attention from 6.483 ms to 1.735 ms, 3.74x.

## Hardware

| Platform | Behavior |
|---|---|
| CPU and MPS | Every CUDA and Triton predicate fails, so everything runs reference |
| NVIDIA CUDA | Triton kernels and eligible Hub and GEMM kernels engage |
| AMD ROCm | Triton can engage, since ROCm wheels ship Triton's AMD backend, but parity is only exercised on NVIDIA in CI |

## Adding an implementation

Call `register()` with a name and a predicate. Out-of-tree compiled kernels can
ship as a separate `libreyolo_kernels` package that registers itself on import,
which keeps a private backend out of the LibreYOLO tree entirely.

Parity is the gate for anything in-tree: an exact forward match against the
reference, and gradients within 1e-6 of the straight-through estimator, over the
shape set the test suite carries.

Kernel selection interacts with [CUDA graphs](/docs/reference/cuda-graphs): the
inference parity matrix ran without the `kernels` package installed, so capture
safety with a compiled kernel active is not covered by it.
