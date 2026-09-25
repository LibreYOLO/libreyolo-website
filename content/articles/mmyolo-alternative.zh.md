---
title: "2026 年 MMYOLO 替代方案：用 LibreYOLO 运行 RTMDet"
description: "MMYOLO 最近一次发布是在 2023 年。通过 LibreYOLO 的直接 API 运行 RTMDet 等受支持的检测器。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLO 已经没人维护了吗？"
    a: "MMYOLO 尚未正式归档，但它最近一次 GitHub 发布是 2023 年 8 月的 0.6.0 版本。"
  - q: "LibreYOLO 能加载任意 MMYOLO 检查点吗？"
    a: "不能。LibreYOLO 支持指定的模型家族，并不承诺可以直接加载任意 MMYOLO 检查点。"
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) 尚未归档，但它最近一次发布是 0.6.0 版本，时间为 2023 年 8 月。如果你只需要 RTMDet 这类受支持的检测器，可能不需要完整的 OpenMMLab 技术栈。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO 并不是完整的 MMYOLO 替代品。它不会复现每一种配置、配方或检查点。它为 RTMDet 和 YOLOX 等受支持的模型家族提供了直接 API，可用于预测、训练、验证和导出。

先运行 `pip install libreyolo`。 [RTMDet 文档](https://www.libreyolo.com/docs/models/rtmdet) 列出了受支持的操作。[RTMDet 简明指南](/articles/rtmdet-without-mmdetection) 介绍了迁移方式。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
