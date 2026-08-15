---
title: "2026 年的 YOLOX 替代方案：用 LibreYOLO 运行"
description: "YOLOX 最新版本发布于 2022 年。通过 LibreYOLO API 运行、训练、验证和导出 YOLOX。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX 是否已被弃置？"
    a: "官方仓库未归档，但最新版本 YOLOX 0.3.0 发布于 2022 年 4 月。称它进入沉寂状态比正式弃置更准确。"
  - q: "LibreYOLO 能训练 YOLOX 吗？"
    a: "可以。LibreYOLO 支持 YOLOX 预测、训练、验证和导出。"
---

YOLOX 仍是一款实用的 Apache-2.0 检测器。它的[官方仓库](https://github.com/Megvii-BaseDetection/YOLOX)并未归档，但最新版本 0.3.0 发布于 2022 年 4 月。维护确实是个问题。模型本身仍然可用。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO 支持从 Nano 到 X 的六种 YOLOX 尺寸，并通过一套 API 提供预测、训练、验证和导出。如果需要原项目精确的 `Exp` 配置流程，请继续使用原项目。

先运行 `pip install libreyolo`。完整 API 见 [YOLOX 文档](https://www.libreyolo.com/docs/models/yolox)。另请参阅简短的 [YOLO-NAS 替代方案](/articles/yolo-nas-with-libreyolo)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
