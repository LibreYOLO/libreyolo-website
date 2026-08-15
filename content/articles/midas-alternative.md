---
title: "MiDaS Alternative in 2026: Depth with LibreYOLO"
description: "The official MiDaS repository is archived. Run its Small and DPT-Large depth models through LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "Is MiDaS abandoned?"
    a: "The official isl-org/MiDaS repository was archived on August 25, 2025."
  - q: "Does MiDaS return metric depth?"
    a: "No. MiDaS returns relative inverse depth with no metric unit or fixed scale between images."
---

The [official MiDaS repository](https://github.com/isl-org/MiDaS) was archived on August 25, 2025. LibreYOLO provides a maintained way to run its Small and DPT-Large checkpoints.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Prediction, zero-shot validation and fixed-resolution export are supported. Training is not. The output is relative inverse depth: higher values are closer, but there is no metric unit or fixed scale between images.

The base install is enough: `pip install libreyolo`. Continue with the [MiDaS docs](https://www.libreyolo.com/docs/models/midas) or the [Real-ESRGAN alternative](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
