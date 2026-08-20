---
title: "YOLO-NAS 仍在维护"
description: "用 LibreYOLO 对 YOLO-NAS 进行预测、训练、验证和导出。SuperGradients 最近一次发版是 2024 年 4 月。我们计划从零训练新权重，以摆脱 Deci 许可证。"
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NAS 已被弃置了吗？"
    a: "SuperGradients 最近一次发版是 2024 年 4 月的 3.7.1。LibreYOLO 在当前 PyTorch 上支持 YOLO-NAS 的预测、训练、验证和导出。"
  - q: "Ultralytics 在维护 YOLO-NAS 吗？"
    a: "Ultralytics 支持推理、验证和导出，不支持训练。"
  - q: "预训练 YOLO-NAS 权重可用于商业用途吗？"
    a: "无论使用哪个库加载，Deci 的预训练权重仍受上游非商业条款约束。从随机初始化开始训练可以避开这些检查点。我们也计划发布从零训练的 COCO 权重。"
---

YOLO-NAS 仍然实用。它原来所在的 [SuperGradients](https://github.com/Deci-AI/super-gradients) 最近一次发版是 2024 年 4 月的 3.7.1。LibreYOLO 为它提供预测、训练、验证和导出。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO 支持检测的 S、M、L 三档，以及姿态估计。检查点从 Deci 的公开 CDN 下载，文件名形如 `LibreYOLONASs.pt`。LibreYOLO 并不托管这些权重，Deci 的非商业条款仍然适用。SuperGradients 把 PyTorch 钉在 1.9 到 1.13；LibreYOLO 要求 2.4 或更高。检测导出覆盖 ONNX、TorchScript、OpenVINO、NCNN、TFLite、Paddle、MNN、ExecuTorch 和 Core AI。不支持 CoreML。

若要在没有 Deci 检查点的情况下训练：

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

我们也计划在 COCO 上从零训练 YOLO-NAS 并发布这些权重。

运行 `pip install libreyolo` 即可安装。模型详情见 [YOLO-NAS 文档](https://www.libreyolo.com/docs/models/yolo-nas)。另有一篇更长的 [SuperGradients 替代方案](/articles/supergradients-alternative) 文章。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
