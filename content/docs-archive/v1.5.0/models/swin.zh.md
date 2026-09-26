---
title: Swin Transformer
families:
  - swin
seo_title: Swin Transformer：用 LibreYOLO 的 LibreSwin 做图像分类
description: 在 LibreYOLO 里对 Swin Transformer 分类器做预测、验证与导出。权重采用 MIT 许可；暂不支持微调。
lead: >-
  Swin Transformer V1：一个分层的 vision transformer，注意力在移位的局部窗口（shifted local
  windows）内计算，而不是在整张图上计算。LibreYOLO 为图像分类提供四种尺寸。
keywords:
  - Swin Transformer
  - swin transformer 图像分类
  - 移位窗口注意力
  - 分层 vision transformer
  - 图像分类 python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data 是一个带 train/ 和 val/ 类别文件夹划分的目录根
        # （ImageFolder 布局），不是数据集 YAML
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀路由，所以导出产物会像任何检查点一样加载，
        # 并返回同一个 Results 对象
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## 安装

Swin 不需要任何可选 extra。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

分类器返回的是 `result.probs` 而不是 `result.boxes`：`top1` 和 `top5` 给出类别索引，`top1conf` 和 `top5conf` 给出它们的置信度。每种尺寸都固定为 224px 输入，因为最后一个注意力阶段就是按这个分辨率搭的；预测、验证和导出只要传入不同的 `imgsz` 都会报错。关于输入源、流式处理和结果处理，见[预测](/docs/predict)。

## 变体

四种尺寸，从 tiny 到 large，由同一套移位窗口堆叠结构搭出来，区别在嵌入向量宽度和各阶段深度。large 在 ImageNet-22k 上预训练，再在 ImageNet-1k 上微调；另外三种直接在 ImageNet-1k 上训练。LibreYOLO 以仅推理的形式收录这个家族：预测、ImageNet 风格的 top-1/top-5 验证和导出都支持，上游的 ImageNet 训练配方没有实现。

## 验证

`val()` 在 ImageFolder 风格的划分上运行（一个带 `train/` 和 `val/` 子目录的目录，每个类别一个文件夹），返回 top-1 和 top-5 精度。

<code-tabs name="val" />

## 导出

<export-matrix />

导出产物会按文件后缀通过 `LibreYOLO()` 重新加载回来，所以一个 `.onnx` 或 `.engine` 文件的表现就像一个检查点（checkpoint），返回同样的 `Results`。[导出](/docs/export)列出了每种格式都接受的参数，以及少数格式额外加上的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的每一个权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
