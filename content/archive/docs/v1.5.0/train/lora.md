---
title: LoRA fine-tuning
seo_title: "LoRA fine-tuning in LibreYOLO"
description: "Fine-tune a transformer detector on low VRAM with lora=True. Which nine families support it, the per-family adapter recipe, and how the checkpoints behave."
lead: "LoRA freezes the pretrained heavy parts of a model and trains small low-rank adapters beside them, plus the layers that must stay dense. In LibreYOLO the whole public interface is one boolean."
keywords:
  - lora fine tuning
  - parameter efficient fine tuning
  - peft
  - dora
  - low vram training
  - rf-detr lora
  - d-fine lora
  - adapter merge
last_verified: "1.5.0"
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: Export merges the adapters
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Merge in place
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
---

## Install

LoRA rides on the optional `peft` dependency.

<code-tabs name="install" />

Without it, `lora=True` raises an `ImportError` naming that command rather than
training a full fine-tune by accident.

## Use it

<code-tabs name="train" />

`lora=True` is the entire interface. Rank, alpha, dropout and target modules are
fixed per family to match each upstream reference, and are not user-facing knobs.

A family that does not support LoRA raises at setup rather than ignoring the
flag:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

The CLI rejects it earlier, before the model is built, using its own allowlist of
the same nine families.

## Which families

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 and v4, EC and ConvNeXt. The gate
is the `supports_lora` attribute on each family's trainer class, and the CLI
carries a matching allowlist.

Task coverage is narrower than family coverage. D-FINE and EC support detection
only, and their segment and pose paths raise. RF-DETR's semantic path raises.
ConvNeXt is classification.

Everything else raises. There is no partial or silent mode.

## What each recipe does

The recipes differ because the architectures differ, and a recipe that works on a
ViT backbone has nothing to attach to on a convolutional one.

RF-DETR uses DoRA, weight-decomposed LoRA, at rank 16 and alpha 16 on the DINOv2
backbone's attention `query`, `key` and `value` projections, matching the RF-DETR
reference. The ViT backbone freezes; the projector, decoder and detection head
keep training normally.

D-FINE, DEIM and RT-DETR v1, v2 and v4 pair a convolutional backbone with a
transformer hybrid encoder and a deformable decoder, so the split moves. The
convolutional backbone freezes entirely, which also skips its backward pass. The
transformer blocks freeze their base weights and train plain LoRA adapters at the
same rank 16 and alpha 16 on their linear layers: the feed-forward `linear1` and
`linear2`, the gate, and the deformable attention projections. Everything else,
the encoder convolution fusion, the input projections, the prediction heads and
the query embeddings, keeps training densely.

Two details in that recipe are deliberate. Decoder self-attention stays frozen
without adapters, because PyTorch's `nn.MultiheadAttention` reads
`out_proj.weight` directly and would silently bypass an injected adapter. And it
is plain LoRA rather than DoRA, because several decoder linear layers are
zero-initialized by design and DoRA's magnitude normalization divides by the
weight norm.

DEIMv2 takes the same recipe with its SwiGLU feed-forward layers `w12` and `w3`
as the targets. Its S, M, L and X sizes also carry a DINOv3 ViT backbone, where
the ViT base freezes and its fused attention `qkv` layers get adapters, while the
Spatial Tuning Adapter convolution pyramid keeps training as the projector
analog. Those `qkv` adapters go in even when the config shipped the ViT frozen,
since adapting a frozen backbone is the point. The sub-S sizes use a
convolutional backbone and take the plain recipe.

EC is a DETR whose backbone is a ViT surrounded by a trainable convolution
projector pyramid. The ViT base freezes and its `qkv` layers get adapters, the
transformer blocks take the shared recipe, and the projector and heads stay
dense.

ConvNeXt blocks carry channels-last linear MLPs, `fc1` and `fc2`, and those take
plain adapters. The depthwise convolutions, the norms and the layer-scale
parameters freeze. The classification head stays dense so custom class counts
keep working.

The detection and classification heads always stay trainable across every recipe,
because a custom class count needs a freshly trained head.

## Checkpoints and export

`best.pt` and `last.pt` keep the adapter tensors, so a LoRA run resumes or gets
inspected like any other. Loading one of those checkpoints needs the `lora`
extra installed, because the loader replays the adapter injection so the keys
line up.

`export()` merges the adapters into dense weights, so an exported artifact
carries no dependency on `peft`. The same merge is available directly for an
in-memory model.

<code-tabs name="merge" />

After a merge the module tree is fully dense and a second merge is a no-op.

## What it saves, and what it does not

LoRA cuts optimizer and gradient memory, and on the families that freeze their
backbone outright it also skips that backbone's backward pass.

Activation memory is unchanged. Forward activations still have to be retained for
whatever remains trainable, and that is usually what sets the peak. For the
tightest VRAM budget, lower `batch` or `imgsz` as well.

## Related

- [Layer freezing](/docs/train/layer-freezing) for the other way to train a
  subset of the weights, which works on every family and needs no extra
  dependency. `freeze` and `lora=True` compose: adapter parameters stay trainable
  even when their parent backbone group is frozen.
- [Hyperparameters](/docs/train/hyperparameters) for `batch`, `imgsz` and the
  rest of `train()`.
