---
title: AlexNet
families: [alexnet]
seo_title: "AlexNet：在 LibreYOLO 里运行经典的 ImageNet 分类器"
description: "用 LibreYOLO 预测、验证并导出 AlexNet。torchvision 权重采用 BSD-3-Clause 许可；微调尚未支持。"
lead: "AlexNet 是赢得 ILSVRC 2012 的卷积网络，它帮助开启了计算机视觉的深度学习时代。LibreYOLO 提供的是这个架构后期的单塔（single-tower）修订版，用于图像分类。"
keywords: [AlexNet, ImageNet, 卷积神经网络, "图像分类 python", "预训练图像分类模型"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreAlexNetb-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data 是一个根目录，下面是按类别分文件夹的 train/ 和 val/ 划分
        # （ImageFolder 布局），而不是数据集 YAML
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## 安装

AlexNet 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

分类器返回的是 `result.probs` 而不是 `result.boxes`：`top1` 和 `top5` 给出类别
索引，`top1conf` 和 `top5conf` 给出对应的置信度。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

只有一种尺寸。随库提供的计算图是 torchvision 发布的后期单塔修订版，第一层有
64 个滤波器，没有局部响应归一化（local response normalization），而不是 2012
年最初的双 GPU 架构。LibreYOLO 只以推理方式提供这个家族：预测、ImageNet 风格的
top-1/top-5 验证和导出都支持，微调没有实现。

## 验证

`val()` 在 ImageFolder 风格的划分上运行（一个带 `train/` 和 `val/` 子文件夹的
目录，每个类别一个文件夹），返回 top-1 和 top-5 精度。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点（checkpoint）一样，返回同样的 `Results`。[导出](/docs/export)列出了每种格式接受的参数，以及其中少数几种额外增加的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>
