---
title: "2026 年的 YOLO-NAS 替代方案：用 LibreYOLO 运行"
description: "无需依赖已经沉寂的 SuperGradients 技术栈即可运行 YOLO-NAS。通过 LibreYOLO 预测、训练、验证和导出。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolo-nas-alternative, object-detection, tutorial]
faq:
  - q: "SuperGradients 仍在维护吗？"
    a: "它并未正式归档，但最新版本是 2024 年 4 月发布的 3.7.1。"
  - q: "预训练 YOLO-NAS 权重可用于商业用途吗？"
    a: "无论使用哪个库加载，Deci 的预训练权重仍受上游非商业条款约束。"
---

YOLO-NAS 仍然实用，但它的原始技术栈已经沉寂。[SuperGradients](https://github.com/Deci-AI/super-gradients) 自 2024 年 4 月发布 3.7.1 版本后就没有再发布新版本。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO 支持 YOLO-NAS 目标检测、姿态估计和旋转框，并提供训练、验证和导出。

运行 `pip install libreyolo` 即可安装。模型详情见 [YOLO-NAS 文档](https://www.libreyolo.com/docs/models/yolo-nas)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
