---
title: U-Net
families:
  - unet
seo_title: U-Net in LibreYOLO
description: >-
  U-Net performs semantic segmentation with an encoder, skip connections and an
  upsampling decoder.
lead: >-
  U-Net performs semantic segmentation with an encoder, skip connections and an
  upsampling decoder.
keywords:
  - U-Net
  - LibreYOLO
  - semantic
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreUNet, SAMPLE_IMAGE

        # Random initialization: train before using predictions.
        model = LibreUNet(size="s", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.semantic_mask)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreUNet


        # Random initialization: train before using predictions.

        model = LibreUNet(size="s", device="cpu")

        # Replace with your semantic segmentation dataset YAML (image/mask pairs).

        model.train(data="path/to/your/semantic_dataset.yaml", pretrained=False,
        epochs=1, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreUNet


        # Replace with a U-Net checkpoint you trained with model.train().

        model = LibreUNet(model_path="path/to/your/unet_checkpoint.pt",
        device="cpu")

        # Replace with your semantic segmentation dataset YAML (image/mask pairs).

        metrics = model.val(data="path/to/your/semantic_dataset.yaml", workers=0)

        print(metrics)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The implementation uses the same-padded S5-D16 graph and an FCN head. No pretrained conversion was verified in the hosted inventory. Construct the class directly to train from scratch.

## Train

Use a semantic dataset with image/mask pairs. Training defaults to 160 epochs, batch 4 and `amp=False`, with a `(512, 1024)` crop and `(1024, 2048)` evaluation canvas. The loss includes an auxiliary head. Export is not supported.

[Dataset setup](/docs/train/datasets) describes the required training data.

<code-tabs name="train" />


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Licensing

<provenance-box></provenance-box>
