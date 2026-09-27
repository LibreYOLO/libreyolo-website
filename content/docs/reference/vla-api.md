---
title: LibreVLA API
seo_title: LibreVLA API | LibreYOLO
description: >-
  LibreVLA provides prediction, training and offline validation for robot action
  policies.
lead: >-
  LibreVLA provides prediction, training and offline validation for robot action
  policies.
keywords:
  - LibreVLA API
  - LibreYOLO
last_verified: 1.6.0
---

## Load

Install `libreyolo[vla]` on Python 3.12 or later. `LibreVLA("smolvla-base")` loads the base policy; `LibreVLA("act")` and `LibreVLA("diffusion")` construct untrained policies. A checkpoint directory is also accepted.

## Predict

`model.predict(source, state=..., instruction=...)` accepts camera frames, frame dictionaries and streams. `cameras=` fixes slot ordering. `set_instruction()` persists a language instruction. Supply the state and camera representations the policy was trained on.

`Results.actions` provides `data`, `first`, `names`, `fps` and `instruction`. Call `reset()` between episodes, especially for Diffusion Policy's observation history.

## Train and validate

`train(data=...)` accepts a LeRobot v3 directory or Hub dataset ID. It returns checkpoint directory paths, including `best`. `val(data=..., split="val", max_batches=...)` measures offline action error. See [robot policies](/docs/tasks/robot-policies) for metrics and [the policy models](/docs/models/librevla) for training differences.

There is no action-policy CLI, tracking or export. Checkpoints contain `libreyolo_vla.json` plus upstream configuration, policy tensors and processors.
