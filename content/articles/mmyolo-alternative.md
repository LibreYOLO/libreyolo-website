---
title: "MMYOLO Alternative in 2026: Run RTMDet with LibreYOLO"
description: "MMYOLO's latest release is from 2023. Run supported detectors such as RTMDet through LibreYOLO's direct API."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "Is MMYOLO abandoned?"
    a: "MMYOLO is not formally archived, but its latest GitHub release is version 0.6.0 from August 2023."
  - q: "Can LibreYOLO load any MMYOLO checkpoint?"
    a: "No. LibreYOLO supports named model families and does not promise drop-in loading of arbitrary MMYOLO checkpoints."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) is not archived, but its latest release, version 0.6.0, dates to August 2023. If you only need a supported detector such as RTMDet, you may not need the full OpenMMLab stack.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO is not a complete MMYOLO replacement. It does not reproduce every config, recipe or checkpoint. It does give supported families such as RTMDet and YOLOX a direct API for prediction, training, validation and export.

Start with `pip install libreyolo`. The [RTMDet docs](https://www.libreyolo.com/docs/models/rtmdet) list the supported operations. The [short RTMDet guide](/articles/rtmdet-without-mmdetection) covers the migration.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
