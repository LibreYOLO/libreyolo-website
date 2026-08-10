---
title: EfficientNetV2
families: [efficientnetv2]
seo_title: "EfficientNetV2：在 Apache-2.0 下训练、验证与导出"
description: "在 LibreYOLO 里用 EfficientNetV2 做图像分类。安装、预测、微调、验证并导出 LibreEfficientNetV2 b0 到 b3。"
lead: "EfficientNetV2 是一个图像分类器，它的深度、宽度以及每个阶段的模块选择都是由神经架构搜索找出来的，同时优化精度和训练速度，而不是只优化精度。LibreYOLO 只支持它的一个任务：分类。"
keywords: [EfficientNetV2, "EfficientNetV2-b0", "图像分类 python", "神经架构搜索", MBConv, "imagenet 分类模型"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEfficientNetV2b0-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientNetV2b0-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=onnx
        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀路由，所以导出产物会像任何检查点一样加载，
        # 并返回同一个 Results 对象
        model = LibreYOLO("LibreEfficientNetV2b0-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## 安装

EfficientNetV2 不需要任何可选 extra。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个模型只是一行的改动。分类器不带检测框，也不带掩码：`result.probs` 保存整图的预测，包含 `top1`、`top5`、`top1conf` 和 `top5conf`。`conf`、`iou` 和 `max_det` 出于 API 一致性会被接受，但不起任何作用，因为在单个概率向量上既没有可以卡阈值的东西，也没有可以抑制的东西。关于输入源、流式处理和结果处理，见[预测](/docs/predict)。

## 变体

四种尺寸，b0 到 b3，每一种都在自己的分辨率和裁剪比例下评测，而不是整个家族共用一个输入尺寸。选哪个尺寸就是一次直接的参数量换精度的取舍。任务是固定的：每种尺寸都只覆盖分类。每种尺寸的权重文件名都以 `-cls.pt` 结尾，工厂读的正是这个后缀，据此路由到这个家族；不需要 `task=` 参数。

## 训练

微调从已发布的 ImageNet 骨干开始，并自动把最后的分类层重建成目标数据集的类别数。除非显式设置，`imgsz` 默认取该尺寸自己的评测分辨率。

<code-tabs name="train" />

不去动它的话，训练器会用 AdamW 跑 100 轮，`lr0=1e-3`，批大小 64，连续 50 轮没有提升就早停。`data` 接受一个数据集根目录（`train/` 和 `val/`，每个类别一个文件夹）、一个像 `imagenette160` 这样的已知短名称，或者一个 `.zip` URL。这里不支持 `lora=True`，传进去会抛异常，因为 LibreYOLO 里的 LoRA 针对的是带 `nn.Linear` 层的 transformer 组件，而这个家族的 MBConv 模块里没有这种层。

关于数据集、数据增强、多卡训练和 logger，见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典。对分类来说就是验证集上的 top-1 和 top-5 精度。

<code-tabs name="val" />

## 导出

<export-matrix />

导出产物会按文件后缀通过 `LibreYOLO()` 重新加载回来，所以一个 `.onnx` 或 `.engine` 文件的表现就像一个检查点，返回同样的 `Results`。[导出](/docs/export)列出了每种格式都接受的参数，以及少数格式额外加上的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的每一个权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>
