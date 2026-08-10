---
title: VGG
families: [vgg]
seo_title: "VGG：在 LibreYOLO 里运行 VGG-16/19 图像分类器"
description: "用 LibreYOLO 预测、验证并导出 VGG 分类器。torchvision 权重采用 BSD-3-Clause 许可；微调尚未支持。"
lead: "VGG 是一个卷积图像分类器，由一叠尺寸统一的 3x3 小卷积堆叠而成，而不是更大的滤波器。LibreYOLO 提供 16 层和 19 层两种尺寸，每种都有普通版和带批归一化（batch normalization）的版本，用于图像分类。"
keywords: [VGG, VGG-16, VGG-19, 卷积神经网络, "图像分类 python", "vgg16 预训练模型"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreVGG16-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data 是一个根目录，下面是按类别分文件夹的 train/ 和 val/ 划分
        # （ImageFolder 布局），而不是数据集 YAML
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## 安装

VGG 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

分类器返回的是 `result.probs` 而不是 `result.boxes`：`top1` 和 `top5` 给出类别
索引，`top1conf` 和 `top5conf` 给出对应的置信度。预测固定以 224px 输入运行，传入
不同的 `imgsz` 会报错。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

四种尺寸：16 层和 19 层卷积，每种各有一个普通版本和一个带批归一化的变体。随库
提供的权重来自 torchvision 后期从零开始的 ImageNet 训练，而不是牛津团队 2014 年
最初 Caffe 版本的转换结果。LibreYOLO 只以推理方式提供这个家族：预测、ImageNet
风格的 top-1/top-5 验证和导出都支持，微调没有实现。

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
