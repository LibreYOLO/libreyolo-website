---
title: 背景移除
seo_title: LibreYOLO 里的背景移除
description: 在 LibreYOLO 里把主体从背景中抠出来。预测一张柔和的 alpha matte，写出一张透明 PNG，再用 MAE 和 S-measure 验证。
lead: 背景移除把主体和它身后的一切分开。LibreYOLO 把它作为 matte 任务提供，为每个像素返回一个柔和的 alpha 值，而不是一张硬的前景掩码。
keywords:
  - 背景移除 python
  - 图像抠图 模型
  - 二分图像分割
  - 透明背景 png 抠图
  - alpha matte
last_verified: 1.5.0
snippets:
  predict:
    - label: 预测一张 matte
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        matte = result.matte

        print(matte.array.shape, matte.array.dtype)   # (H, W) float32，取值在 [0,
        1]
    - label: 写出一张透明 PNG
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() 会把源图和作为 alpha 通道的 matte 合成起来
        result.save("subject.png")

        rgba = result.cutout()   # 内存里同一个 (H, W, 4) uint8 数组
        print(rgba.shape)
    - label: 合成到新的背景上
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # 白色

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # 一个包含 images/ 和一个 matte 目录的目录
        # 可以代替数据集 YAML
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # 越低越好
        print(metrics["metrics/Smeasure"])   # fitness，越高越好
  export:
    - label: 导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: 运行导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出产物的加载方式和普通检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## 定义

`matte` 任务从一张 RGB 图像出发，为每个像素预测一个 alpha 值：`1` 表示完全前景，
`0` 表示完全背景。这个值是连续的而不是二值的，这正是这个任务的意义所在。硬掩码只差
一个阈值，在 0.5 处就能得到；而柔和的 matte 还额外保留了头发、毛发和运动模糊
边缘处的部分覆盖，这些正是二值掩码会丢掉的东西。

一次预测会填充 `result.matte`，这是一个 `Matte` 载荷，里面是原图画布上取值在
`[0, 1]` 之间的 `(H, W)` float32 数组，通过 `.array` 以 NumPy 的形式取用。
`result.cutout()` 会把源图和这条 alpha 合成为一个 `(H, W, 4)` uint8 RGBA 数组，
`result.save(path)` 则把同样的东西写成一张透明背景的 PNG。`result.boxes` 始终为空，
所以 `conf`、`iou` 和 `max_det` 都不起作用。

## 模型

有两个家族支持 `matte`，而且它们共用一条前向路径。

[BiRefNet](/docs/models/birefnet) 是这个任务据以构建的双边参考（bilateral-reference）
网络，这里发布为一个 Swin-L 档的检查点（checkpoint）。

[FeyNobg](/docs/models/feynobg) 是 Feyn Inc. 加深后的变体：在 BiRefNet 的架构上，
把第三个 Swin 阶段从 18 个块扩到 24 个块，然后重新训练。LibreYOLO 为它复用了
BiRefNet 的前向路径、预处理和单 logit 输出，所以预测、验证和检查点处理的行为完全
一致；权重和家族身份则是 FeyNobg 自己的。

两者的权重许可不同。两个许可都写在各自的模型页上，而具体那个检查点在 Hugging Face
仓库上标注的许可才是权威的。

## 预测

权重会在首次使用时从 Hugging Face 下载，并缓存到本地。

<code-tabs name="predict" />

两个家族都固定在原生的 1024x1024 画布上运行，再把 matte 缩放回原图。不支持其他
分辨率，因为 Swin 骨干的相对位置表和这个尺寸绑在一起，尺寸对不上时它会把这些表
插值得很糟，而不是直接报错。`Results.save()` 只为 matte 结果定义，并且需要源图；
除非你自己传一张进去，否则它会从 `Results.path` 重新加载。输入源、流式处理和结果
处理见[预测](/docs/predict)。

## 数据集格式

matte 验证会把每张 RGB 图像和一张文件名主干相同的单通道真值（ground truth）alpha
matte 配对，其中 0 是背景，255 是前景。

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

把这个根目录作为 `data=` 传进去就够了：matte 目录会在 `mattes/`、`matte/`、`gt/`、
`masks/`、`mask/` 和 `alpha/` 之间自动识别。另一种做法是用数据集 YAML，用 `path`
加上 `val_images` 和 `val_mattes` 指定相对于它的目录：

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` 和 `names` 只是 schema 占位符；matte 模型返回的是 `Results.matte`，不是检测
结果。matte 的取值会除以 255，作为 `[0, 1]` 之间的 alpha 读入；形状和预测画布不一致
的 matte 会用双线性插值缩放到匹配。完整约定见[数据集格式](/docs/reference/dataset-formats)。

## 训练

两个 matte 家族都没有训练实现：在它们上面调用 `train()` 都会抛出
`NotImplementedError`，matte 支持只覆盖预测、验证和导出。每个模型页都会指明提供训练
代码的上游项目，以及把检查点转回来的转换脚本。

## 验证

`val()` 走的是模型自己的 `predict`，所以验证用的是这个家族确切的预处理，两个指标都在
原图画布上计算。

<code-tabs name="val" />

`metrics/MAE` 是相对真值 alpha 的平均绝对误差，取值在 `[0, 1]` 之间，越低越好。
`metrics/Smeasure` 是 Fan 等人的 S-measure（ICCV 2017），一种结构相似度，它会奖励
把主体的形状和它的空洞都还原对，而这正是单看逐像素平均会漏掉的；越高越好。S-measure
同时也是 `fitness`，也就是最佳检查点选择读取的那个数。两个指标都与分辨率无关。

## 导出

导出的 matte 模型会按文件后缀通过 `LibreYOLO()` 加载回来，所以这个产物的表现和检查点
一样，返回同样的 `Results`。

<code-tabs name="export" />

TorchScript 是这个任务已经验证过的路径。ONNX 转换能跑通，但还没有达到同样的一致性
标准，其余格式则不可用。每种格式的覆盖情况见 [BiRefNet](/docs/models/birefnet) 和
[FeyNobg](/docs/models/feynobg) 页面，以及[完整导出矩阵](/docs/reference/export-matrix)。
