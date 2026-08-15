---
title: "YOLO-NAS Alternative in 2026: Run It with LibreYOLO"
description: "Run YOLO-NAS without building on a quiet SuperGradients stack. Predict, train, validate and export through LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolo-nas-alternative, object-detection, tutorial]
faq:
  - q: "Is SuperGradients still maintained?"
    a: "It is not formally archived, but its latest release is 3.7.1 from April 2024."
  - q: "Can pretrained YOLO-NAS weights be used commercially?"
    a: "Deci's pretrained weights retain their upstream non-commercial terms, whichever library loads them."
---

YOLO-NAS remains useful, but its original home is quiet. [SuperGradients](https://github.com/Deci-AI/super-gradients) has not shipped a release since version 3.7.1 in April 2024.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO supports YOLO-NAS detection, pose and oriented boxes, plus training, validation and export.

Install with `pip install libreyolo`. The [YOLO-NAS docs](https://www.libreyolo.com/docs/models/yolo-nas) have the model details.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
