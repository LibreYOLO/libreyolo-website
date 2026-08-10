---
title: ViT
families: [vit]
seo_title: "ViT：在 LibreYOLO 里运行经典的 Vision Transformer 分类器"
description: "在 LibreYOLO 里预测、验证和导出 ViT 分类器。采用 Apache-2.0 许可的 AugReg 权重；暂不支持微调。"
lead: "经典的 Vision Transformer：一个纯 transformer，作用在固定大小的图像 patch 上，带一个学习得到的 class token，不含卷积。LibreYOLO 收录了四种 AugReg 预训练尺寸，用于图像分类。"
keywords: [ViT, Vision Transformer, AugReg, "图像分类 python", "vision transformer 图像分类"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreViTti-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data 是一个目录根，下面是 train/ 和 val/ 两个按类别分文件夹的划分
        # 也就是 ImageFolder 布局，而不是数据集 YAML
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## 安装

ViT 不需要任何可选 extra。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

分类器返回的是 `result.probs` 而不是 `result.boxes`：`top1` 和 `top5` 给出类别
索引，`top1conf` 和 `top5conf` 给出对应的置信度。预处理会缩放并中心裁剪到固定的
224px 输入，用的是 timm 的 AugReg 评估配方：双三次插值，裁剪比例 0.9。数据源、
流式处理和结果处理见 [预测](/docs/predict)。

## 变体

四种尺寸，从 tiny 到 large，共用同一张固定 224px、patch-16 的计算图，区别在于
嵌入向量宽度和 transformer 深度。LibreYOLO 以仅推理的形式收录这个家族：预测、
ImageNet 风格的 top-1/top-5 验证和导出都支持，AugReg 的微调配方没有实现。

## 验证

`val()` 跑在 ImageFolder 风格的划分上——一个带 `train/` 和 `val/` 子文件夹的
目录，每个类别一个文件夹——返回 top-1 和 top-5 精度。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点（checkpoint）一样，返回同样的 `Results`。[导出](/docs/export)
列出了每种格式都接受的参数，以及其中少数几种额外加的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
