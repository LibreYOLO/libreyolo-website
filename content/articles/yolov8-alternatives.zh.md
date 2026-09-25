---
title: "YOLOv8 替代方案推荐：2026 年值得关注的模型"
description: "RF100-VL 衡量检测器在 COCO 之外数据集上的微调表现。RF-DETR 和 YOLO-NAS 的得分高于 YOLOv8。了解如何在 LibreYOLO 中运行这两个模型。"
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "2026 年最好的 YOLOv8 替代方案是什么？"
    a: "如果使用 GPU，推荐 RF-DETR-S。它在 LibreYOLO 的 RF100-VL 测试中取得了 60.41 mAP50-95。RF100-VL 论文报告 YOLOv8m 的得分为 56.9。卷积模型可以选 YOLO-NAS-S，得分为 58.00。"
  - q: "RF-DETR 的迁移能力比 YOLOv8 更好吗？"
    a: "在 RF100-VL 上是的。LibreYOLO 测得 RF-DETR-S 为 60.41，RF-DETR-M 为 61.13。Roboflow 报告的得分分别为 60.2 和 61.2。论文报告 YOLOv8m 的得分为 56.9。"
  - q: "YOLO-NAS 可以用于商业用途吗？"
    a: "Deci 的预训练权重不可商用。其架构采用 Apache-2.0 许可。LibreYOLO 不提供这些权重。许可证详情请看 YOLO-NAS 文章。"
---

选择检测器通常会看 COCO 表现。也可以换个角度，看看模型在其他数据上的微调效果。[RF100-VL](https://arxiv.org/abs/2505.20612) 就是衡量这一点的基准：在 100 个数据集上，分别用 COCO 预训练的检查点训练 100 轮，然后在每个数据集的测试集上评分。最终数字是平均 mAP50-95。

根据[论文附录](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf)，YOLOv8m 的得分为 56.9。我们运行的两个模型得分更高：RF-DETR 和 YOLO-NAS。原始运行结果见 [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results)。

## RF-DETR

在 [LibreYOLO 的测试](/articles/rf100vl-benchmark)中，RF-DETR-S 得分为 60.41，RF-DETR-M 得分为 61.13。[Roboflow 报告](https://rfdetr.roboflow.com/latest/learn/benchmarks/)的得分分别为 60.2 和 61.2。Nano 到 Large 版本均采用 Apache-2.0 许可。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[文档](/docs/models/rf-detr) | [权重](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [运行记录 `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S 得分为 58.00。如果你想要 YOLO 风格的 CNN，它是一个卷积模型选项。Deci 发布的权重不可商用。LibreYOLO 不托管这些权重。详情见：[YOLO-NAS 仍在持续维护](/articles/yolo-nas-with-libreyolo)。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[文档](/docs/models/yolo-nas) | [运行记录 `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

安装 RF-DETR 时运行 `pip install "libreyolo[rfdetr]"`，安装 YOLO-NAS 时运行 `pip install libreyolo`。测试页面：[RF100-VL](/articles/rf100vl-benchmark)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [文档](https://www.libreyolo.com/docs)
