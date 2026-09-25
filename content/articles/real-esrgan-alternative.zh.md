---
title: "Real-ESRGAN 替代方案：用 LibreYOLO 放大图像（2026）"
description: "通过 LibreYOLO 运行 Real-ESRGAN 2x 和 4x 图像放大，包括适用于大图的分块推理。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN 已经停止维护了吗？"
    a: "该仓库尚未归档，但其最新 GitHub release 是 2022 年 9 月发布的 0.3.0 版本。"
  - q: "LibreYOLO 能训练 Real-ESRGAN 吗？"
    a: "不能。LibreYOLO 支持 Real-ESRGAN 的预测、验证和导出，不支持训练。"
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 并未正式归档，但其最新 GitHub release 是 0.3.0 版本，发布于 2022 年 9 月。LibreYOLO 通过常规预测 API 运行它的 2x 和 4x 检查点。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO 支持 2x、4x 和快速 4x 检查点，也支持分块预测、PSNR/SSIM 验证和导出。目前尚未实现训练。上游的去噪强度混合功能也不在这个移植版本中。

运行 `pip install libreyolo` 即可试用。完整的图像修复选项见 [Real-ESRGAN 文档](https://www.libreyolo.com/docs/models/real-esrgan)。有关深度估计，请参阅 [MiDaS 替代方案](/articles/midas-alternative)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
