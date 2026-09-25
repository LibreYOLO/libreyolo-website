---
title: SuperGradients 替代方案：用 LibreYOLO 运行 YOLO-NAS 并继续训练
description: "英伟达于 2024 年收购 Deci 后，SuperGradients 停止发布新版本。LibreYOLO 是持续维护的替代方案，可运行相同的 YOLO-NAS 权重，并通过一个 MIT 许可的 API 完成预测、训练和导出。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "SuperGradients 还在维护吗？"
    a: "没有。最后一个版本是 2024 年 4 月发布的 3.7.1，时间就在英伟达收购 Deci 之前。此后仓库没有发布新版本，问题无人回复，文档网站也已下线。它没有正式归档，但已经无人维护。"
  - q: "SuperGradients 最好的替代方案是什么？"
    a: "如果要运行 YOLO-NAS，LibreYOLO 会通过持续维护且采用 MIT 许可的 API 加载相同的 Deci 权重，并支持训练、验证和导出。如果你依赖 SuperGradients 为其他架构提供的训练配置，固定其依赖版本后，旧仓库仍然可以运行。"
  - q: "我可以将 YOLO-NAS 用于商业用途吗？"
    a: "无论使用哪个库加载，Deci 的预训练权重都不能用于商业用途。架构本身采用宽松许可，因此你可以用自己的数据从头训练 YOLO-NAS，得到一个历史中不含 Deci 检查点的模型。"
  - q: "LibreYOLO 能完全替代 SuperGradients 吗？"
    a: "不能完全替代。LibreYOLO 不能为 YOLO-NAS 导出 CoreML，而且该系列的 TensorRT 路径尚未验证；SuperGradients 的量化感知训练配置也没有直接对应方案。遇到这些需求时，请保留旧仓库。"
---

SuperGradients 是 Deci 的开源训练库，也是 YOLO-NAS 的所在地。YOLO-NAS 是迄今发布的精度最高的实时检测器之一。2024 年 5 月，英伟达收购了 Deci，SuperGradients 随后便没了动静。最后一个版本 3.7.1 于 2024 年 4 月 8 日发布。supergradients.com 上的文档已经下线。大约 120 个问题仍处于未解决状态，其中一个问题的标题是[「这个项目是不是已经没人维护了？」](https://github.com/Deci-AI/super-gradients/issues/2062)，维护者对此没有回应，而这本身就是一种回答。

仓库没有归档，所以乍看之下似乎还在维护。实际安装时，你马上就会感受到它已经被弃置：`super-gradients` 固定依赖 `torchmetrics==0.8`，这与当前的 PyTorch 技术栈冲突，还会一并引入 hydra、omegaconf、boto3 和 tensorboard。每过一个月，这些固定版本就会更难与其余环境兼容，而且不会有修复。

这并不意味着 YOLO-NAS 模型变差了。大型变体在 COCO 上仍能达到 52.2 mAP，同时保持实时速度。模型本身没问题，只是它的家已经烧毁了。

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## 在 LibreYOLO 中运行相同的权重

LibreYOLO 会直接从 Deci 的 CDN 加载 YOLO-NAS 检查点，并使用适用于所有其他模型家族的同一套 API。你不需要编写 hydra 配置，也不需要再拆开一层预测封装：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # 首次运行时自动下载
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

S、M 和 L 三种检测变体都能运行，姿态模型也可以：换成 `LibreYOLONASs-pose.pt`，就能得到 COCO 关键点。所有模型家族都返回相同的 `Results` 对象，因此要在自己的数据上比较 YOLO-NAS 与 RF-DETR 或 D-FINE，只需改一行代码，无需另建一套代码库。

## 不止推理，还能训练和导出

SuperGradients 首先是一个训练库，所以真正的替代方案必须支持训练。LibreYOLO 使用与其他地方相同的调用方式，在你的数据集上微调 YOLO-NAS：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

你也可以从随机初始化的模型开始训练，这一点与许可有关（下文详述）。验证会针对任何符合你格式的数据集返回 mAP 指标，导出则支持 ONNX、TorchScript、OpenVINO、NCNN 和 TFLite。这套导出支持范围比原始库曾为 YOLO-NAS 提供的更广。完整详情见 [YOLO-NAS 文档页](https://www.libreyolo.com/docs/models/yolo-nas)，另有一篇简短说明介绍了 [YOLO-NAS 仍在持续维护](/articles/yolo-nas-with-libreyolo)。

## 旧仓库仍然适用的场景

坦诚地说，以下三种情况下，你应该继续安装 SuperGradients：

**CoreML。** LibreYOLO 不能将 YOLO-NAS 导出为 CoreML。如果你要在 Apple 设备上发布产品，并且需要 `.mlpackage`，SuperGradients 仍有可用的路径。

**TensorRT。** SuperGradients 为 YOLO-NAS 记录并测试了 TensorRT 流程，包括批大小方面的细节。LibreYOLO 对这个模型家族的 TensorRT 支持尚未验证。

**量化感知训练配置。** YOLO-NAS 专为 INT8 设计，SuperGradients 提供的 QAT 配置让它在这方面表现出色。LibreYOLO 支持以 INT8 和 FP16 量化导出，但没有对应的训练配置。

如果你把环境固定在 2024 年左右的版本，仓库仍然可以运行。只是不要再基于一个无人维护的基础构建新东西。

## 关于权重的说明

预训练 YOLO-NAS 权重属于 Deci，以非商业许可发布；无论哪个库加载这些权重，该许可都随之适用。LibreYOLO 不托管也不镜像这些权重；下载直接来自 Deci 的公共 CDN，并会在开始前显示 Deci 的条款。用于研究和非商业用途时，两种方式都可以。

如果你需要用于商业用途的 YOLO-NAS，现在有一条清晰的路径：架构本身采用宽松许可，因此用自己的数据从头训练，就能得到一个没有派生自任何 Deci 检查点的模型。LibreYOLO 正好支持通过 `LibreYOLONAS(None, size="s")` 来实现这一点。

## 试试看

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO 采用 MIT 许可，可在 Linux、Mac 和 Windows 上运行，并且在 GPU、Apple Silicon 和普通 CPU 上都无需修改代码即可使用。一套 API 覆盖 YOLO-NAS、RF-DETR、D-FINE、DEIM、YOLOX、RTMDet 等更多模型，支持目标检测、分割、姿态、分类、深度和跟踪等任务。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://www.libreyolo.com/docs)
