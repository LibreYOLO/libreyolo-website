---
title: DeiT
families:
  - deit
seo_title: DeiT 图像分类器：预测、验证与导出
description: >-
  在 LibreYOLO 里运行 DeiT 图像分类器：一个冻结的、仅推理的博物馆家族，提供 tiny、small 和 base 三种尺寸，采用
  Apache-2.0 许可。
lead: >-
  DeiT（Data-efficient image Transformer）是一个纯粹的 Vision Transformer 分类器，只在
  ImageNet-1k 上训练，没有用额外的预训练数据。LibreYOLO 以冻结、仅推理的展品形式收录 tiny、small 和 base 三种
  patch-16 尺寸。
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - 图像分类 python
  - ImageNet
  - deit 预训练模型
  - vision transformer 图像分类
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## 安装

DeiT 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

这个家族只提供推理：`train()` 会抛出 `NotImplementedError`，所以本页没有训练一节。
预测、验证和导出都支持。权重在首次使用时从 Hugging Face 下载，并缓存在本地。文件名
里的 `-cls` 后缀是必需的，它选择分类任务。

<code-tabs name="predict" />

返回的 `Results` 对象带的是 `probs` 张量而不是 `boxes`；`top1` 和 `top5` 索引
ImageNet-1k 的 1,000 个类别，`top1conf` 是排在第一的那个预测的 softmax 分数。每种
尺寸都有一个由位置嵌入向量决定的固定输入分辨率：预处理会缩放并中心裁剪到这个分辨率，
传入不同的 `imgsz` 会直接报错，而不是悄悄重采样。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 验证

`val()` 返回一个带 top-1 和 top-5 精度的字典，在按惯例的 `train/<class>/` 和
`val/<class>/` 文件夹结构组织的数据集上测得。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点（checkpoint）一样，返回同样的 `Results`。在不装 LibreYOLO 的裸
运行时里跑这张计算图也是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
