---
title: "YOLOX Alternative in 2026: Run It with LibreYOLO"
description: "The latest YOLOX release is from 2022. Run, train, validate and export YOLOX through the LibreYOLO API."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "Is YOLOX abandoned?"
    a: "The official repository is not archived, but its latest release is YOLOX 0.3.0 from April 2022. Quiet is more accurate than formally abandoned."
  - q: "Can LibreYOLO train YOLOX?"
    a: "Yes. LibreYOLO supports YOLOX prediction, training, validation and export."
---

YOLOX is still a useful Apache-2.0 detector. Its [official repository](https://github.com/Megvii-BaseDetection/YOLOX) is not archived, but the latest release, 0.3.0, dates to April 2022. The maintenance problem is real. The model still works.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO supports six YOLOX sizes, from Nano to X, with prediction, training, validation and export behind one API. Use the original project if you need its exact `Exp` configuration workflow.

Start with `pip install libreyolo`. The [YOLOX docs](https://www.libreyolo.com/docs/models/yolox) cover the full API. There is also a short [YOLO-NAS alternative](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
