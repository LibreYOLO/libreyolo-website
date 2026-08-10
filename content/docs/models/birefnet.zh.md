---
title: BiRefNet
families: [birefnet]
seo_title: "BiRefNet：LibreYOLO 里的背景移除与抠图"
description: "在 LibreYOLO 里用 BiRefNet 做背景移除和二分图像分割。安装、预测、验证并导出通用检查点。"
lead: "一种双边参考（bilateral-reference）网络，预测一张把主体和背景分开的柔和 alpha matte。LibreYOLO 为 BiRefNet 的 matte 任务提供推理和验证。"
keywords: [BiRefNet, "birefnet 抠图", "背景移除 python", "图像抠图 模型", "alpha matte", "二分图像分割", "透明背景 png python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreBiRefNetl-matte.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 抠图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8：源图 RGB 加上作为 alpha 通道的 matte
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # 一个包含 images/ 和自动识别出的 matte 目录
        # （mattes/、matte/、gt/、masks/、mask/ 或 alpha/）的目录
        # 也可以代替数据集 YAML
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出产物的加载方式和普通检查点
        # 一样，返回的也是同一个 Results 对象
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
---

## 安装

BiRefNet 不需要任何可选的额外依赖。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重会在首次使用时从 Hugging Face 下载，并缓存到本地。

<code-tabs name="predict" />

matte 结果不带检测框；`result.matte` 是一个取值在 `[0, 1]` 之间的稠密
`(H, W)` float32 数组，1 表示完全前景，0 表示完全背景。和二值掩码不同，柔和的
matte 会保留头发、毛发这类抗锯齿的边缘细节。`result.cutout()` 会把源图和这条
alpha 通道合成为一个 RGBA 数组，`result.save(path)`（或者在预测调用上加
`save=True`）会直接把它写成一张透明背景的 PNG。模型固定在原生的 1024x1024
画布上运行；不支持其他分辨率，因为 Swin 骨干的相对位置表和它绑在一起，尺寸对不上时
它会把这些表插值得很糟，而不是直接报错。输入源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

只发布了一个检查点（checkpoint）：`l`，也就是 Swin-L 档的 BiRefNet-general
模型，上游默认的质量选择。这个家族的代码还支持一个 Swin-T 轻量档 `t`，但目前还
没有发布对应的 LibreYOLO 转换版本。

## 验证

`val()` 会在成对的 image/matte 目录上报告两个指标，两者都在 `[0, 1]` 之间，
并且与分辨率无关：MAE，即相对真值（ground truth）alpha 的平均绝对误差（越低越
好）；以及 S-measure（Fan 等，ICCV 2017），一种结构相似度，它会奖励保留主体形状
和空洞的结果，而这正是单看像素 MAE 会漏掉的（越高越好）。验证走的是模型自己的
`predict`，因此用的是这个家族确切的预处理。

<code-tabs name="val" />

验证只做推理；微调是记录在案的后续计划，而不是已经发布的功能（未来任何训练器都会
继承的那条分辨率限制，见「预测」一节）。

## 导出

<export-matrix />

导出产物按文件后缀通过 `LibreYOLO()` 加载回来，所以一个 `.onnx` 文件的表现和检查点
一样，返回同样的 `Results`。TorchScript 是已经验证过的路径；ONNX 转换能跑通，但
还没有达到同样的一致性标准。[导出](/docs/export)列出了每种格式接受的参数，以及少
数几种格式额外增加的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可

<provenance-box></provenance-box>

## 引用

<citation-block />
