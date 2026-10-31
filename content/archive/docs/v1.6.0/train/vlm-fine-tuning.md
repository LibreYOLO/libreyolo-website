---
title: VLM detection fine-tuning
seo_title: VLM detection fine-tuning | LibreYOLO
description: Qwen3-VL fine-tuning trains language-model adapters against detection targets.
lead: Qwen3-VL fine-tuning trains language-model adapters against detection targets.
keywords:
  - VLM detection fine-tuning
  - LibreYOLO
last_verified: 1.6.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("qwen3-vl-2b", device="cpu")
        model.train(data="coco8", epochs=1, batch=1, workers=0)
---

## Install

```bash
pip install "libreyolo[vlm-train]"
```

## Train

<code-tabs name="train" />

LoRA is the default. Full language-model tuning is optional; the vision tower stays frozen. The default learning rate is 1e-4 for LoRA or 2e-5 for full tuning. Defaults are 10 epochs, batch 1, accumulation 8, workers 0, horizontal-flip probability 0.5 and gradient checkpointing enabled.

Use a standard detection dataset YAML with .txt label files; its class names become the prompt vocabulary. COCO-JSON annotations are not supported. This workflow has its own training arguments and checkpoint format; detector [LoRA](/docs/train/lora) uses a different trainer.

## Reload

Training writes checkpoint directories under an incremented `runs/vlm/train`. Load a returned checkpoint directory with `LibreVLM(path)`. Validation loss selects the best checkpoint, or training loss when the dataset has no val split. Optimizer-state resume and detection-mAP validation are not supported.
