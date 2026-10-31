---
title: LibreVLA
families:
  - smolvla
seo_title: LibreVLA in LibreYOLO
description: >-
  LibreVLA runs robot policies from camera frames and robot state and returns
  action chunks.
lead: >-
  LibreVLA runs robot policies from camera frames and robot state and returns
  action chunks.
keywords:
  - LibreVLA
  - LibreYOLO
  - act
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreVLA, SAMPLE_IMAGE


        # Python 3.12 or later. This demonstrates shapes with synthetic state.

        model = LibreVLA("smolvla-base", device="cpu")

        result = model.predict(SAMPLE_IMAGE, state=[0.0] * 6, instruction="pick
        up the object")

        print(result.actions.data.shape)

        print(result.actions.first)
---

## Install

```bash
pip install "libreyolo[vla]"
```

## Predict

<code-tabs name="predict" />

The extra requires Python 3.12 or later. Supply real camera observations and the state representation used during training for meaningful actions. The library returns action values; the caller owns the robot control loop. Call `reset()` between episodes.

## Train

### SmolVLA

`LibreVLA()` selects SmolVLA. Training updates its action expert with the vision backbone frozen. Pass a LeRobot dataset ID or local v3 directory to `train(data=...)`. Defaults are 5 epochs, batch 8, workers 0 and `val_split=0.1`.

### ACT

`LibreVLA("act")` constructs an untrained action policy. Train it on a LeRobot dataset before prediction. It uses the dataset camera, state and action shapes without a language instruction. Its image backbone may download separately.

### Diffusion Policy

`LibreVLA("diffusion")` also starts without pretrained policy weights. It keeps observation history between calls; `reset()` clears that history between episodes.

## Validate

`val(data=...)` reports `val/action_l1`, `val/action_mse`, `val/action_l1_first` and per-dimension `val/action_l1_dims`. These compare predicted and recorded actions offline; they are not robot task success rates.

## Checkpoints

Training saves directories containing policy weights, processors and `libreyolo_vla.json`. Reload a returned best-checkpoint directory with `LibreVLA(path)`. See the [policy API](/docs/reference/vla-api).
## Licensing

<provenance-box></provenance-box>
