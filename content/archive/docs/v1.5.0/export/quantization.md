---
title: Quantization
seo_title: "Quantize a LibreYOLO model in PyTorch"
description: "LibreYOLO's PyTorch quantization API: nine recipes, calibration kept apart from training data, QAT and QAD, and two deployment artifacts."
lead: "Quantization in LibreYOLO runs entirely in PyTorch: model.quantize() swaps a model's Conv2d and Linear modules for quantized equivalents and calibrates them. The result keeps the ordinary predict, val, train and save contract, so a quantized model is scored by the same validators as a float one."
keywords:
  - libreyolo quantization
  - int8 ptq
  - quantization aware training
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - calibration dataset
  - qdq onnx export
last_verified: "1.5.0"
meta:
  - label: Call
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Command
    value: "libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml"
    mono: true
  - label: Extra
    value: "None. Quantization runs in PyTorch."
  - label: Families
    value: "yolo9, rfdetr, birefnet, feynobg"
  - label: Recipes
    value: "fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2"
    mono: true
  - label: Deployment artifacts
    value: 'export(format="pt") for a packed checkpoint, export(format="onnx") for a QDQ INT8 graph'
    mono: true
verification: "Read from libreyolo/quant/api.py, libreyolo/models/base/model.py, libreyolo/cli/commands/quantize.py and docs/quantization.md on the dev branch. The checkpoint size figures are the measured values recorded in docs/quantization.md."
snippets:
  quantize:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Structure swap plus calibration. calib is a small UNLABELED image set,
        # read forward-only to derive activation ranges and scales.
        qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

        print(qmodel.quant_info())
        qmodel.val(data="coco8.yaml")          # same validators as a float model
        qmodel.save("LibreYOLO9s-int8.pt")     # checkpoint carries a quant manifest
    - label: CLI
      language: bash
      code: |
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco128.yaml
    - label: Arguments
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # data.yaml path or built-in name; None skips calibration
            samples=128,               # maximum calibration images
            batch=8,                   # calibration batch size
            algorithm="auto",          # auto and minmax are the same; percentile is the alternative
            keep_high_precision=None,  # None uses the family policy
            verbose=True,
        )
  reload:
    - label: A quantized checkpoint reloads as one
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The quant manifest rebuilds the quantized structure and scales
        # before the weights are loaded.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT is plain train() on a quantized model
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # A finetune, not a from-scratch run: use finetune learning rates.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD adds the existing distillation arguments
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5 --lr0 1e-4
  export:
    - label: Packed PyTorch checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Writes LibreYOLO9s-int8-final.pt: packed low-bit weights and scales,
        # fp32 masters stripped, the non-quantized remainder cast to fp16.
        qmodel.export(format="pt")

        # remainder="fp32" keeps the non-quantized tensors exact.
        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # In-graph QuantizeLinear/DequantizeLinear pairs carrying the model's
        # own calibrated or QAT-trained scales.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: Back to float, keeping QAT-trained weights
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # Any float exporter now applies, at any precision it supports.
        qmodel.export(format="tensorrt", half=True)
---

## Install

Quantization needs no extra. The module swap, the calibration pass and the
simulated arithmetic all run in PyTorch, so `pip install libreyolo` is the whole
requirement. The deployment artifacts need whatever their own format needs, which
for the ONNX path is `libreyolo[onnx]`.

## Quantize

<code-tabs name="quantize" />

`quantize()` transforms the loaded model in place and returns it. No gradients are
involved: the swap installs quantized modules and the calibration pass runs
forward only.

The resulting checkpoint is an ordinary LibreYOLO checkpoint with a `quant`
manifest attached, so it reloads with its structure and scales intact:

<code-tabs name="reload" />

Trainer checkpoints written during a QAT run carry the manifest too, which means
`best.pt` from such a run is itself a quantized checkpoint.

## Recipes

Four families are supported: `yolo9`, `rfdetr`, `birefnet` and `feynobg`.

| Recipe | What it does | Families | Calibration |
|---|---|---|---|
| `fp16` | Cast to half precision with a float32 input and output contract. Inference only. | all four | none |
| `bf16` | Cast to bfloat16, which keeps float32's exponent range. The fix when fp16 overflows on a DETR-style model. Inference only. | all four | none |
| `fp8` | E4M3 weights and activations on `Conv2d` and `Linear`: per-channel weight scales, calibrated per-tensor activation scales. | all four | required |
| `int8` | W8A8 on `Conv2d` and `Linear`: per-channel symmetric weights, per-tensor affine activations. | all four | required, or `calib=None` for weights only |
| `w4a16` | Grouped symmetric INT4 weights, group 128 along `in_features`, float activations, on `Linear`. | rfdetr, birefnet, feynobg | not needed |
| `w4a8` | Grouped INT4 weights plus calibrated INT8 activations, on `Linear`. | rfdetr, birefnet, feynobg | required |
| `nvfp4` | W4A4 NVFP4 on `Linear`: E2M1 elements, 16-element blocks, FP8 E4M3 block scales, FP32 tensor scale. Dynamic activation scaling. | rfdetr, birefnet, feynobg | not needed |
| `mxfp4` | OCP MXFP4 on `Linear`: E2M1 elements, 32-element blocks, power-of-two E8M0 block scales. Dynamic activation scaling. | rfdetr, birefnet, feynobg | not needed |
| `int2` | Research only: grouped 2-bit weights, group 64, plus INT8 activations, on `Linear`. Post-training alone is unusable, so QAT or QAD is required. | rfdetr | required |

The sub-8-bit recipes target `nn.Linear` and are rejected for `yolo9` on purpose:
that acceleration is GEMM only on current hardware, so convolutions stay in higher
precision. YOLO9 uses `int8` or `fp8`. `int2` is rejected for `birefnet` and
`feynobg` because those families are inference only, so the QAT healing the recipe
depends on is unavailable there.

Per-family defaults keep the first layer and the heads in float, and the YOLO9 DFL
convolution is never quantized: it is a fixed integral-expectation operator.
Override with `keep_high_precision=("head.",)` when you have a reason to.

## Calibration data is not training data

`calib=` takes a few hundred images, reads no labels, and runs forward only to
estimate activation ranges. `data=` in `train()` and `val()` is the labeled
dataset used for gradients and metrics. They are different arguments with
different purposes, and the default for `calib` is `coco128.yaml`.

`algorithm="minmax"` keeps the absolute extremes seen across calibration batches
and is what `"auto"` selects. `"percentile"` uses the mean of per-batch 0.1 and
99.9 percentiles; it was measured to collapse DETR-family accuracy, because
transformer activation outliers are load-bearing. What actually fixes small-model
INT8 sensitivity is calibrating on enough batches: with the `coco128` default,
YOLO9-t lands within about one mAP point of its float score. The chosen algorithm
is recorded in the checkpoint manifest.

## Recover accuracy

<code-tabs name="train" />

Quantized modules keep fp32 master weights and apply fake quantization with a
straight-through estimator, so gradients reach the masters and the existing
trainers work unchanged: EMA, AMP, checkpoint resume and the distillation
arguments all compose.

QAT is a finetune of an already-trained model. Use finetune learning rates rather
than the from-scratch defaults, or a short run will destroy the pretrained weights
regardless of quantization. QAD availability follows family distillation support,
which today means `yolo9` and `rfdetr`.

`fp16`- and `bf16`-quantized models are inference only, and the trainer rejects
them with a pointer to `amp=True`.

## Export

<code-tabs name="export" />

`format="pt"` crystallizes the model. Packed low-bit weights and scales replace the
masters, and the non-quantized remainder is cast to fp16 unless
`remainder="fp32"` is passed. The packing invariant is that unpacking reproduces
the simulation bit for bit on the device you finalized on, so the finalized file
scores exactly what you validated. Measured: YOLO9-s int8 goes from 29.5 MB to
9.6 MB, RF-DETR-n nvfp4 from 122 MB to 26 MB. Loading one gives an
inference-ready model, and calling `train()` on it reconstructs the masters from
the packed weights automatically.

`format="onnx"` applies to `int8` models and emits a QDQ graph carrying the
model's own calibrated or QAT-trained scales, which ONNX Runtime and TensorRT run
with real INT8 kernels. This is a different path from
[`export(format="onnx", int8=True)`](/docs/export/onnx) on a float model, where
ONNX Runtime derives the scales itself.

The cast recipes need no quantized exporter at all:

<code-tabs name="dequantize" />

## Constraints

Quantized arithmetic executes in simulation, which is fake quantization computed
in float32 islands even under AMP. Simulation is numerics-true, so a `val()` score
on any device is a real claim about the quantized arithmetic. It is not a speed
claim.

Two exceptions execute natively. `fp16` and `bf16` are ordinary casts. Finalized
`fp8` modules run their GEMM directly on packed E4M3 weights through
`torch._scaled_mm` on Ada, Hopper and Blackwell class hardware, using the same
calibrated activation scales as the simulation; setting `LIBREYOLO_KERNELS=off`
restores the exact simulated path everywhere.

Deployment coverage is narrower than the recipe list. Only `int8` has a
deployable ONNX form here; `fp8` and the sub-8-bit linear recipes execute in
PyTorch and crystallize through `format="pt"`. Requesting an ONNX export from
them raises with that instruction, as does requesting any non-ONNX format from an
`int8` model: build downstream engines from the QDQ graph instead.

Exporting an `int8` model whose activations were never calibrated logs a warning
and produces a graph carrying weight quantization only.
