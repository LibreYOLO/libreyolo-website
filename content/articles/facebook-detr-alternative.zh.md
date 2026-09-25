---
title: "Facebook DETR 替代方案：2026 年如何通过 LibreYOLO 运行"
description: "原始 Facebook Research DETR 仓库已归档。通过 LibreYOLO 运行四种官方 DETR 变体。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR 已经无人维护了吗？"
    a: "原始的 facebookresearch/detr 仓库于 2024 年 3 月 12 日归档。"
  - q: "LibreYOLO 能训练原版 DETR 吗？"
    a: "不能。LibreYOLO 支持 DETR 预测、验证和导出，但不支持其 500 轮训练方案。"
---

原始的 [Facebook Research DETR 仓库](https://github.com/facebookresearch/detr) 于 2024 年 3 月 12 日归档。LibreYOLO 通过与其他模型家族相同的 API，让这个原始检测器仍可使用。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

提供四种基于 ResNet-50 和 ResNet-101 的变体，可用于预测、验证，以及导出为 ONNX 和 TorchScript。尚未实现训练。LibreYOLO 还使用固定的 800 × 800 输入尺寸，而不是复现原始的保持宽高比评估流水线。

运行 `pip install libreyolo` 后即可试用。[DETR 文档](https://www.libreyolo.com/docs/models/detr) 列出了全部四个检查点。若要与 CNN 对比，请参阅 [EfficientDet 替代方案](/articles/efficientdet-alternative)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
