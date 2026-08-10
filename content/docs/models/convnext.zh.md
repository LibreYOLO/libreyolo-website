---
title: ConvNeXt
families: [convnext]
seo_title: "ConvNeXt：在 Apache-2.0 下训练、验证与导出"
description: "在 LibreYOLO 里用 ConvNeXt 做图像分类。安装、预测、用 LoRA 微调、验证并导出 LibreConvNeXt tiny/small/base。"
lead: "ConvNeXt 是一个完全由标准卷积搭出来的图像分类器，从 ResNet 出发，一个模块一个模块地朝 vision transformer 的设计选择改造。LibreYOLO 只支持它的一个任务：分类。"
keywords: [ConvNeXt, "ConvNeXt tiny", "图像分类 python", "纯卷积网络", "imagenet 预训练分类模型"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀路由，所以导出产物会像任何检查点一样加载，
        # 并返回同一个 Results 对象
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## 安装

ConvNeXt 不需要任何可选 extra。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

用 `lora=True` 做适配器微调是个例外，需要 `lora` 这个 extra。

```bash
pip install "libreyolo[lora]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个模型只是一行的改动。分类器不带检测框，也不带掩码：`result.probs` 保存整图的预测，包含 `top1`、`top5`、`top1conf` 和 `top5conf`。`conf`、`iou` 和 `max_det` 出于 API 一致性会被接受，但不起任何作用，因为在单个概率向量上既没有可以卡阈值的东西，也没有可以抑制的东西。关于输入源、流式处理和结果处理，见[预测](/docs/predict)。

## 变体

三种尺寸，tiny/small/base，训练和评测方式完全一样，所以选哪个就是一次直接的参数量换精度的取舍。任务是固定的：每种尺寸都只覆盖分类。每种尺寸的权重文件名都以 `-cls.pt` 结尾，工厂读的正是这个后缀，据此路由到这个家族；不需要 `task=` 参数。

## 训练

微调从已发布的 ImageNet 骨干开始，并自动把最后的分类层重建成目标数据集的类别数。

<code-tabs name="train" />

不去动它的话，训练器会用 AdamW 跑 100 轮，`lr0=1e-3`，批大小 64，连续 50 轮没有提升就早停。`data` 接受一个数据集根目录（`train/` 和 `val/`，每个类别一个文件夹）、一个像 `imagenette160` 这样的已知短名称，或者一个 `.zip` URL。ConvNeXt 的模块里带着 LoRA 需要的 `nn.Linear` MLP，所以这里支持 `lora=True`，它会把适配器注入模块的 MLP，而不是微调整个骨干。

关于数据集、数据增强、多卡训练和 logger，见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典。对分类来说就是验证集上的 top-1 和 top-5 精度。

<code-tabs name="val" />

## 导出

<export-matrix />

导出产物会按文件后缀通过 `LibreYOLO()` 重新加载回来，所以一个 `.onnx` 或 `.engine` 文件的表现就像一个检查点，返回同样的 `Results`。[导出](/docs/export)列出了每种格式都接受的参数，以及少数格式额外加上的参数。

## 检查点

这个家族已发布的每一个权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

这个家族只提供 ConvNeXt V1。ConvNeXt-V2 的小尺寸预训练检查点采用 CC-BY-NC-4.0 许可，被刻意排除在外，因为非商用权重没法在一个 MIT/商用库里重新分发。

</provenance-box>

## 引用

<citation-block />
