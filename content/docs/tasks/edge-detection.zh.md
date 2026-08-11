---
title: 边缘检测
seo_title: LibreYOLO 中的边缘检测
description: 在 LibreYOLO 里从一张图像预测稠密的边缘概率图。转换一个检查点，对概率图取阈值，用 ODS 和 OIS 验证，然后导出。
lead: 边缘检测预测的是每个像素落在物体边界上的可能性有多大。LibreYOLO 把它做成 edge 任务，返回的是原图画布上的一张稠密概率图，而不是一组线段。
keywords:
  - 边缘检测 python
  - 图像边缘检测 深度学习
  - 边缘概率图
  - ODS OIS F-measure
  - dexined teed 边缘检测
last_verified: 1.5.0
snippets:
  predict:
    - label: 预测一张边缘图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreYOLO 不附带任何边缘检查点，先按下文转换一个
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) float32，取值在 [0, 1]
        print(edges.binary(0.5).sum())    # 阈值 0.5 下的边缘像素数
    - label: 自己选阈值
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # 连续的概率图会被保留下来，阈值始终由你来定
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: 保存可视化结果
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() 会把概率图画出来，边缘结果和法线结果都有定义
        result.plot().save("edges.png")
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: 修改扫描范围和匹配容差
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: 导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: 运行导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出产物的加载方式和检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## 定义

`edge` 任务从一张 RGB 图像出发，为每个像素预测一个概率：`0` 表示不是边缘，`1`
表示是边缘。概率图保持连续，所以把它变成二值边界图的阈值由调用方自己选，而合适的
阈值取决于数据集和下游用途。

一次预测会填充 `result.edges`，它是一个 `EdgeMap` 载荷，里面是原图画布上取值在
`[0, 1]` 的 `(H, W)` float32 数组。`.array` 以 NumPy 形式返回这张图，
`.binary(threshold)` 返回一张布尔掩码。`result.boxes` 保持为空，所以 `conf`、`iou`
和 `max_det` 都不起作用。`Results.plot()` 覆盖了这个任务，会直接把概率图画出来。

## 模型

有三个家族支持 `edge`。

[DexiNed](/docs/models/dexined)，也就是 Dense Extreme Inception Network，把多个侧
输出融合成一张概率图，原生分辨率是 352 px。

[TEED](/docs/models/teed)，也就是 Tiny and Efficient Edge Detector，是一个原生分辨
率同样为 352 px 的小网络，下采样步长是 4，而 DexiNed 是 16，所以它能接受更多的
`imgsz` 取值。

[LibreMODUS](/docs/models/libremodus) 把 Canny 风格的边缘作为一个 any-to-any 模型
的其中一个输出目标。它需要 `modus` 这个额外依赖和你自己已认证的 Hugging Face 账号，
而且既不提供 `val()` 也不提供 `export()`，所以它不参与下面的验证和导出部分。

## 预测

LibreYOLO 不发布任何边缘检查点（checkpoint）。官方发布的 DexiNed 和 TEED 权重是在
BIPED 上训练的，而 BIPED 公布的数据集条款把用途限制在非商业目的，所以 LibreYOLO 不
镜像它们。转换一个你有权使用的检查点，然后按路径加载转换后的文件：

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

文件名必须带上 `-edge` 这个任务后缀，加载器才能识别它。`imgsz` 必须能被网络的下采样
步长整除，不满足时 LibreYOLO 会抛出一个明确指出除数的错误。输入源、流式处理和结果
处理见[预测](/docs/predict)。

## 数据集格式

边缘验证把每张 RGB 图像和一张同名、同分辨率的单通道图配成一对，另外还可以带一张可选
的有效性掩码。

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

目标是一张单通道的 PNG 或 TIF，而不是 RGB 可视化图。整数图会除以其 dtype 的最大值；
浮点图本身就必须是有限值并且落在 `[0, 1]` 里。掩码像素非零时算作有效，填充像素永远
不计入指标。`edge_invert: true` 覆盖那些把边缘存成白底黑线的数据源。完整的约定见
[数据集格式](/docs/reference/dataset-formats)。

## 训练

LibreYOLO 里没有任何一个边缘家族实现了训练：三个家族的 `train()` 都会抛出
`NotImplementedError`。每个模型页面都会给出对应的转换脚本，把在别处训练的检查点转成
LibreYOLO 能加载的文件。

## 验证

`val()` 报告 BSDS 风格的 F-measure。连续的预测先用四方向梯度非极大值抑制做细化，然后
预测的边缘像素和真值（ground truth）边缘像素在一个距离容差内一对一匹配。

<code-tabs name="val" />

`metrics/ODS` 是数据集最优尺度下的 F-measure：在每个阈值上把整个数据集的匹配计数汇总
起来，再报告这些汇总 F-measure 里最好的一个。它同时也是 `fitness`，即最优检查点选择
读取的那个数字。`metrics/OIS` 是单图最优尺度下的 F-measure，也就是每张图各自最好的
F-measure 在所有图上的平均值，所以它让每张图都能挑自己的阈值。`metrics/best_threshold`
是产生 ODS 的那个唯一阈值，也就是推理时应该在 `edges.binary()` 里复用的那个。

有两个参数决定这次扫描。`edge_thresholds` 是要试的阈值集合，默认从 0.01 到 0.99，以
百分之一为步长。`edge_max_dist` 是匹配容差，按图像对角线的比例给出，默认 `0.0075`；
相距超过这个值的一对像素不算匹配。

## 导出

导出的边缘模型按文件后缀通过 `LibreYOLO()` 加载回来，所以一个 `.onnx` 文件的表现和
检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

边缘导出使用固定分辨率、batch 为 1 的运行时约定：`dynamic` 以及 1 以外的 `batch` 都
会被拒绝，导出的图只输出一张融合后的概率图。分格式的支持情况在
[DexiNed](/docs/models/dexined) 和 [TEED](/docs/models/teed) 页面上，以及
[完整的导出矩阵](/docs/reference/export-matrix)里。[导出](/docs/export)列出了每种
格式接受的参数。
