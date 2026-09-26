---
title: Custom training fitness
seo_title: Custom training fitness | LibreYOLO
description: >-
  A fitness callback selects best checkpoints and controls patience with a
  custom scalar score.
lead: >-
  A fitness callback selects best checkpoints and controls patience with a
  custom scalar score.
keywords:
  - Custom training fitness
  - LibreYOLO
last_verified: 1.6.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        class RecallFitness:
            def fitness(self, metrics):
                return float(metrics["metrics/recall(B)"])

        model = LibreYOLO("LibreYOLO9s.pt", device="cpu")

        model.train(data="coco8.yaml", epochs=1, workers=0,
        device="cpu", callbacks=RecallFitness())
---

## Define the scorer

<code-tabs name="train" />

Implement `fitness(metrics)` on an object passed through `callbacks=`. The input is the read-only scalar mapping used by `TrainEpochEvent.val_metrics`. Return one finite real number; higher is better. A callback list may contain only one scorer. VLM and VLA fine-tuning do not accept a fitness callback.

## Checkpoint selection

Scoring runs on rank zero for each validation result used for selection, before checkpoint writing and epoch callbacks. It changes best-checkpoint selection and patience without replacing the original metrics.

Checkpoints record `fitness_source="callback"`. Callback code and state are not serialized, so custom-fitness training cannot resume. Start a new run from the saved weights instead.

See [validation](/docs/train/validation) for the metric keys available to a scorer.
