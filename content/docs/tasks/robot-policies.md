---
title: Robot policies
seo_title: Robot policies | LibreYOLO
description: >-
  Robot policies return an action chunk from camera observations and robot
  state.
lead: >-
  Robot policies return an action chunk from camera observations and robot
  state.
keywords:
  - Robot policies
  - LibreYOLO
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreVLA, SAMPLE_IMAGE


        # Python 3.12+. Synthetic observation for checking the API.

        model = LibreVLA(device="cpu")

        result = model(SAMPLE_IMAGE, state=[0.0] * 6, instruction="pick up the
        object")

        print(result.actions.data.shape)
---

## Models

[LibreVLA](/docs/models/librevla) provides SmolVLA, ACT and Diffusion Policy. SmolVLA conditions on a language instruction. ACT and Diffusion Policy train without a pretrained policy base.

## Predict

<code-tabs name="predict" />

Use real robot state and camera frames for meaningful predictions. `Actions.data` is a float32 array of shape `(T, D)`; `Actions.first` is the first action. The payload also stores action names, control rate and instruction when available. Call `reset()` between episodes.

## Dataset format

Use a LeRobot v3 dataset directory or Hub dataset ID. Training reads episode boundaries, camera features, state, actions and timestamps from that dataset. There is no detection-style YAML for this task.

## Train

Install `libreyolo[vla]` on Python 3.12 or later. Call `model.train(data=...)`; defaults are 5 epochs, batch 8 and a 0.1 held-out episode fraction.

## Validate

Offline validation reports `val/action_l1`, `val/action_mse`, `val/action_l1_first` and per-dimension `val/action_l1_dims`. These are errors against recorded actions in dataset units. They do not measure task success on a robot.

## Export

Policy export is not supported. Reload a training checkpoint directory through `LibreVLA(path)`.
