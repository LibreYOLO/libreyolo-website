---
title: FeyNobg
families:
  - feynobg
seo_title: FeyNobg：LibreYOLO 里的背景移除
description: >-
  在 LibreYOLO 里用 FeyNobg 做背景移除和 alpha 抠图，它是 Feyn Inc. 加深并重新训练过的 BiRefNet
  变体。安装、预测并验证。
lead: >-
  一个来自 Feyn Inc. 的背景移除模型，它加深了 BiRefNet 的架构并重新训练。LibreYOLO 为 FeyNobg 的 matte
  任务提供推理和验证。
keywords:
  - FeyNobg
  - feynobg 背景移除
  - 背景移除 python
  - 图像抠图 模型
  - alpha matte
  - 二分图像分割
  - 透明背景 png python
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 抠图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8：源图 RGB 加上作为 alpha 通道的 matte
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # 一个包含 images/ 和自动识别出的 matte 目录
        # （mattes/、matte/、gt/、masks/、mask/ 或 alpha/）的目录
        # 也可以代替数据集 YAML
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## 安装

FeyNobg 不需要任何可选的额外依赖。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

## 预测

检查点（checkpoint）会在首次使用时从 Hugging Face 上的 LibreYOLO 组织下载，并
缓存到本地，和其他家族一样，只是它还没有列进本页的检查点表格。

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

只发布了一个尺寸 `l`，采用 Swin-L 档的骨干。FeyNobg 拿来 BiRefNet 的架构，把它
的第三个 Swin 阶段从 18 个 block 加深到 24 个再重新训练，所以 LibreYOLO 的移植
复用了 BiRefNet 的前向路径、预处理和单 logit 输出约定；预测、验证和检查点处理的
行为都和 `birefnet` 家族一样。

## 验证

`val()` 会在成对的 image/matte 目录上报告两个指标，两者都在 `[0, 1]` 之间，
并且与分辨率无关：MAE，即相对真值（ground truth）alpha 的平均绝对误差（越低越
好）；以及 S-measure（Fan 等，ICCV 2017），一种结构相似度，它会奖励保留主体形状
和空洞的结果，而这正是单看像素 MAE 会漏掉的（越高越好）。验证走的是模型自己的
`predict`，因此用的是这个家族确切的预处理。

<code-tabs name="val" />

验证只做推理。上游的 `nobg` 库提供了采用 Apache-2.0 许可的训练代码；今天要微调，
意味着在那边训练，再用 LibreYOLO 自己的转换脚本转换结果，而不是在这个家族上调用
`train()`——它会直接报错，而不是跑一个不完整的训练器。

## 许可

<provenance-box></provenance-box>

## 引用

<citation-block />
