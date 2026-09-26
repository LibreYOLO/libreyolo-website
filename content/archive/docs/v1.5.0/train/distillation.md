---
title: Knowledge distillation
seo_title: "Knowledge distillation in LibreYOLO"
description: "Train a small detector against a larger teacher or a frozen DINOv2 backbone: the MGD, CWD and feature-MSE losses, tap points, and family support."
lead: "Distillation adds a second loss term that pulls the student's intermediate feature maps toward a frozen teacher's. LibreYOLO taps features with forward hooks, so the teacher's own head and loss are never involved."
keywords:
  - knowledge distillation
  - masked generative distillation
  - channel-wise distillation
  - feature distillation
  - dinov2 teacher
  - teacher student training
  - mgd loss
  - cwd loss
last_verified: "1.5.0"
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A larger checkpoint of the same family supervises the small one.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A frozen self-supervised ViT supervises one backbone stage.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Tuning the loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # global distillation weight
            distill_tau=1.0,   # CWD softmax temperature
        )
---

## Distill from a larger checkpoint

Setting `distill_model` turns distillation on. The value is a teacher checkpoint,
loaded through the same factory as any other model.

<code-tabs name="detector" />

The teacher runs forward under `no_grad`, and under autocast when AMP is on, so
the frozen model does not pay full-precision compute at every step. Forward hooks
capture its feature maps at named tap points, the loss compares them with the
student's, and the result is added to the training loss and reported as a
component named `distill`.

## Distill from a frozen foundation backbone

A self-supervised ViT can supervise a single student backbone stage instead. The
teacher's features come from its own feature extractor rather than hooks, and the
loss handles the mismatch between a patch grid and a convolutional stride.

<code-tabs name="foundation" />

`distill_model` recognizes `dinov2`, which is DINOv2-base, plus `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`, `dinov2-large`,
and any raw hub id starting with `facebook/dinov2`. Anything else is treated as a
teacher checkpoint path.

This path uses `feat_mse` regardless of `distill_loss_type`, and needs
`transformers` installed. A teacher that loads with missing weight keys aborts
rather than distilling against a partly random backbone.

## Which families

Distillation support is a method on the student model, and there are two of them.

`get_distill_config()` provides the multi-scale tap points a detector teacher
supervises. YOLOv9, YOLOX and RF-DETR implement it.

`get_backbone_distill_config()` provides the single backbone stage a foundation
teacher supervises. YOLOv9 implements it, and it is the only family that does.

Anything else raises rather than training without the loss:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Tap points

The tap points are fixed per family and per role, so teacher and student do not
need to be the same architecture; they need matching feature strides.

| Family | Role | Tap points | Strides |
|---|---|---|---|
| YOLOv9 | teacher or student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | foundation student | `backbone.elan3` | 16 |
| YOLOX | teacher or student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher or student | `model.backbone.0.projector.stages.0` | probed at setup |

Mismatched strides raise before training starts:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

That check is skipped for foundation teachers, whose whole point is that the
grids differ.

## The three losses

`distill_loss_type` selects the feature loss for a detector teacher. A foundation
teacher always uses `feat_mse`.

`mgd`, masked generative distillation, masks a fraction of the student's spatial
positions and trains a small two-convolution generator to reconstruct the
teacher's full feature map from what remains. `distill_mask_ratio` sets the
masked fraction, default 0.65.

`cwd`, channel-wise distillation, turns each channel's spatial activations into a
probability distribution and minimizes the KL divergence channel by channel.
`distill_tau` is the softmax temperature, default 1.0.

`feat_mse` aligns the student's channels to the teacher's with a 1x1 convolution,
resizes the teacher's grid to the student's bilinearly, and takes the mean
squared error. `distill_normalize=True` L2-normalizes both feature maps over the
channel dimension first, which makes the match angle-only and scale-invariant. It
defaults to `False`.

`dis` is the global weight applied on top. Left unset, each loss uses its own
published default: 2e-5 for MGD, 1.0 for CWD and 1.0 for feature MSE. Those
differ by five orders of magnitude, so a weight tuned for one loss type is
meaningless for another.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` and `distill_normalize` have no CLI flags.
They are Python arguments or `cfg=` YAML keys. RF-DETR is also Python-only for
distillation as a whole, because its CLI argument mapping does not carry the
distillation keys.

## Adapters, checkpoints and multi-GPU

Every loss builds small trainable modules that live outside the student: the 1x1
channel adapters, and MGD's generator. They get their own optimizer parameter
group at the run's effective learning rate.

Those modules are written into the checkpoint under a `distiller` key and
restored on resume, so a resumed run does not restart its projectors cold.

Under DDP the adapters sit outside the wrapped student, which means the DDP
reducer never sees their gradients. The trainer all-reduces them explicitly each
step, so every rank trains the same adapters.

CUDA graph capture is not available on a distillation run. Passing
`cuda_graph=True` logs one line and trains eager. See
[Training performance](/docs/train/performance).

## Related

- [Layer freezing](/docs/train/layer-freezing) and
  [LoRA fine-tuning](/docs/train/lora), neither of which is blocked from being
  combined with distillation.
- [Hyperparameters](/docs/train/hyperparameters) for the rest of `train()`.
