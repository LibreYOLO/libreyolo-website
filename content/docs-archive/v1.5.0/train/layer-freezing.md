---
title: Layer freezing
seo_title: "Freeze layers during training in LibreYOLO"
description: "Freeze part of a model for transfer learning: an integer count of family freeze groups, an explicit index list, or module and parameter name selectors."
lead: "Freezing holds selected weights fixed while the rest of the model trains. Selectors address a family's own ordered freeze groups or its module names, not raw layer numbers from a YAML graph."
keywords:
  - freeze layers
  - transfer learning
  - freeze backbone
  - frozen batchnorm
  - freeze groups
  - fine tune head only
last_verified: "1.5.0"
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # The first 10 groups are the whole YOLOv9 backbone.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: By name
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Several selectors
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: List a family's freeze groups in order
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
---

## Freeze something

`freeze` is optional and defaults to no freezing.

<code-tabs name="train" />

Freezing runs after the model is built and after any head rebuild for a new class
count, and before the optimizer is created, so the optimizer only ever receives
trainable parameters.

## What a selector can be

| Value | Meaning |
|---|---|
| `None`, `False`, `""`, `"none"` | Train every parameter |
| `10` or `"10"` | Freeze the first ten family freeze groups |
| `[0, 3, 7]` | Freeze those zero-based groups |
| `"backbone"` | Freeze the matching group, module or parameter prefix |
| `["backbone", "neck"]` | Freeze each listed selector |
| `["backbone", 3]` | Mixed lists work |

A string is parsed before it is interpreted, so the CLI and a YAML config accept
the same shapes as Python. `freeze="[0, 3, 'head']"` is parsed as a literal list,
`freeze="backbone,neck"` splits on the comma, and a bare decimal string becomes a
count.

`freeze=True` is rejected as ambiguous.

Name selectors match a freeze group name, a module name, or a parameter-name
prefix, and glob characters `*`, `?` and `[` work. A leading `model.` is treated
flexibly, so `backbone` and `model.backbone` both hit whichever spelling the
family uses internally.

## Groups are family-defined

An integer addresses a family's own ordered list of freeze groups, not a position
in a shared graph. LibreYOLO's families are not all one YAML-indexed sequential
model, so a raw layer number would mean something different on each of them.

YOLOv9 orders its groups from the input side: ten backbone stages, then six neck
stages, then the head. That is why `freeze=10` is exactly the backbone.
`backbone`, `neck` and `head` are stable name selectors on top of it.

RF-DETR's groups are `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output` and `head`. Names are the better choice
here, because transformer components do not map onto a layer count. `backbone`
matches both backbone groups by prefix.

Families that do not define semantic groups fall back to a conservative default:
each direct child of the model that owns at least one parameter, in declaration
order. That is usually a short list, so a large integer will not find enough
groups:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

To see the real list rather than guessing:

<code-tabs name="groups" />

## Failures are loud

Every way of getting this wrong raises rather than training something you did not
ask for.

A selector that matches nothing raises, naming the selectors that missed:

```text
freeze selector(s) matched no parameters: 'backbon'
```

A freeze that would leave nothing trainable raises, both at freeze time and again
when the optimizer is built:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Which is what `freeze="all"` does, since `all` matches every parameter.

When freezing succeeds, one line records what happened:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## Frozen BatchNorm stops updating

A frozen parameter still sits inside a module whose running statistics would keep
moving. Every BatchNorm-style module whose parameters land in the frozen set is
switched to eval mode, and the trainer re-applies that after each epoch's
`model.train()` call, so the statistics stay fixed for the whole run.

This is on by default and is what makes freezing a backbone actually freeze it.

## Composing with LoRA

`freeze` and `lora=True` work together. On RF-DETR, DEIM and ConvNeXt the
adapter parameters are preserved as trainable even when their parent group is
frozen, which is the combination you want: a frozen backbone with adapters
learning on top of it. See [LoRA fine-tuning](/docs/train/lora).

## Scope

This is static freezing decided at startup. Scheduled unfreezing and progressive
freezing are not part of the interface.

## Related

- [Hyperparameters](/docs/train/hyperparameters) for the rest of `train()`.
- [Distillation](/docs/train/distillation) for the other way to move a large
  model's knowledge into a training run.
