---
title: "Facebook DETR Alternative in 2026: Run It with LibreYOLO"
description: "The original Facebook Research DETR repository is archived. Run four official DETR variants through LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Is Facebook DETR abandoned?"
    a: "The original facebookresearch/detr repository was archived on March 12, 2024."
  - q: "Can LibreYOLO train the original DETR?"
    a: "No. LibreYOLO supports DETR prediction, validation and export, but not its 500-epoch training recipe."
---

The original [Facebook Research DETR repository](https://github.com/facebookresearch/detr) was archived on March 12, 2024. LibreYOLO keeps the original detector usable through the same API as its other model families.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Four ResNet-50 and ResNet-101 variants are available for prediction, validation, ONNX and TorchScript export. Training is not implemented. LibreYOLO also uses a fixed 800 by 800 input, rather than reproducing the original aspect-preserving evaluation pipeline.

Try it after `pip install libreyolo`. The [DETR docs](https://www.libreyolo.com/docs/models/detr) list all four checkpoints. For a CNN comparison, see the [EfficientDet alternative](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
