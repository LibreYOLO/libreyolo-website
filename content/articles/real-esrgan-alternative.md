---
title: "Real-ESRGAN Alternative in 2026: Upscale with LibreYOLO"
description: "Run Real-ESRGAN 2x and 4x image upscaling through LibreYOLO, including tiled inference for large images."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Is Real-ESRGAN abandoned?"
    a: "The repository is not archived, but its latest GitHub release is version 0.3.0 from September 2022."
  - q: "Can LibreYOLO train Real-ESRGAN?"
    a: "No. LibreYOLO supports Real-ESRGAN prediction, validation and export, not training."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) is not formally archived, but its latest GitHub release, version 0.3.0, dates to September 2022. LibreYOLO runs its 2x and 4x checkpoints through the regular prediction API.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO supports 2x, 4x and fast 4x checkpoints, with tiled prediction, PSNR/SSIM validation and export. Training is not implemented. The upstream denoise-strength blend is also outside this port.

Use `pip install libreyolo` to try it. Full restore options are in the [Real-ESRGAN docs](https://www.libreyolo.com/docs/models/real-esrgan). For depth, see the [MiDaS alternative](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
