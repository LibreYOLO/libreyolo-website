---
title: "MiDaS 替代方案：用 LibreYOLO 运行深度估计模型"
description: "官方 MiDaS 仓库已归档。通过 LibreYOLO 运行其 Small 和 DPT-Large 深度模型。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "MiDaS 已经停止维护了吗？"
    a: "官方 isl-org/MiDaS 仓库已于 2025 年 8 月 25 日归档。"
  - q: "MiDaS 会返回度量深度吗？"
    a: "不会。MiDaS 返回相对逆深度，不带度量单位，不同图像之间也没有固定尺度。"
---

[MiDaS 官方仓库](https://github.com/isl-org/MiDaS) 已于 2025 年 8 月 25 日归档。LibreYOLO 提供了一种持续维护的方式来运行其 Small 和 DPT-Large 检查点（checkpoint）。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

支持预测、零样本验证和固定分辨率导出。不支持训练。输出是相对逆深度：数值越大表示距离越近，但没有度量单位，不同图像之间也没有固定尺度。

基础安装即可：`pip install libreyolo`。接下来可以阅读 [MiDaS 文档](https://www.libreyolo.com/docs/models/midas)，或查看 [Real-ESRGAN 替代方案](/articles/real-esrgan-alternative)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
