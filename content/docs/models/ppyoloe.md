---
title: PP-YOLOE
families:
  - ppyoloe
seo_title: 'PP-YOLOE: prediction and training in LibreYOLO'
description: >-
  PP-YOLOE is an anchor-free object detector with detection training and
  class-head resizing.
lead: >-
  PP-YOLOE is an anchor-free object detector with detection training and
  class-head resizing.
keywords:
  - PP-YOLOE
  - LibreYOLO
  - detect
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePPYOLOE, SAMPLE_IMAGE


        model = LibrePPYOLOE(model_path=input("Path to upstream weights: "),
        device="cpu")

        result = model(SAMPLE_IMAGE)

        print(result.boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePPYOLOE


        model = LibrePPYOLOE(model_path=input("Path to upstream weights: "),
        device="cpu")

        # Enter the path to your dataset YAML or classification folder.

        model.train(data=input("Dataset path: "), epochs=1, device="cpu",
        workers=0)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

Released weights download from the source CDN with a pinned SHA-256 check. The constructor also accepts a local upstream checkpoint. The four sizes are s, m, l and x.

## Train

Training uses ATSS assignment before switching to task-aligned assignment. `static_assigner_epochs` defaults to 30% of the requested epochs. A new class count rebuilds the class-prediction layers. AMP defaults to `False`.

[Dataset setup](/docs/train/datasets) describes the required training data.

<code-tabs name="train" />

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Licensing

<provenance-box></provenance-box>
