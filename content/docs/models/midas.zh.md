---
title: MiDaS
families: [midas]
seo_title: "MiDaS：LibreYOLO 里的单目深度估计"
description: "在 LibreYOLO 里用 MiDaS 做单目深度估计。安装、预测、验证并导出两个采用 MIT 许可的变体，权重从 isl-org 下载。"
lead: "MiDaS 是单目相对深度估计，在混合数据集上用尺度和偏移不变的损失函数训练，正是这条工作线确立了后来各家族沿用的零样本深度迁移流程。LibreYOLO 支持它的 depth 任务：预测和零样本验证，没有训练这一步。"
keywords: [MiDaS, 单目深度估计, DPT, 相对深度, "深度图 python", "零样本深度估计"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 本地还没有这个文件：LibreYOLO 会从官方的 isl-org/MiDaS GitHub release
        # 下载，并在使用前校验固定的 SHA-256
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMiDaSl-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Small 变体
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # EfficientNet-Lite3 编码器，比 DPT-Large 的 l 尺寸更小更快
        model = LibreYOLO("LibreMiDaSs-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## 安装

MiDaS 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

MiDaS 是唯一一个 LibreYOLO 没有在自己的 Hugging Face 组织下重新发布的深度家
族。按 LibreYOLO 的文件名请求一个检查点（checkpoint），会直接从 `isl-org/MiDaS`
的 GitHub releases 下载对应的官方文件，校验固定的 SHA-256，并在首次使用前给它加
上 LibreYOLO 的检查点元数据；之后的运行会复用缓存的本地文件。原因见许可证一节。

<code-tabs name="predict" />

`result.depth_map` 带的是一张稠密的相对逆深度图：值越大表示离相机越近，而且这些
值没有度量单位，也没有跨图像的统一尺度。`save=True` 会把这张图经过色彩映射的可
视化结果写入磁盘；`Results.plot()` 不覆盖这个家族，因为它只为表面法线和边缘定
义。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

两个变体用的是不同的编码器，而不只是同一个编码器的不同尺度。`s` 是 MiDaS v2.1
Small，一个 EfficientNet-Lite3 编码器。`l` 是 DPT-Large，一个 ViT-L/16 编码器，
配上 MiDaS 为稠密预测引入的 DPT 解码器。两者的预处理也不一样：`s` 用的是带上界
的保持宽高比缩放，加 ImageNet 的均值/标准差归一化，`l` 用的是最小的保持宽高比缩
放，均值和标准差都取 0.5。想要更轻的 CNN 就选 `s`，想要 transformer 解码器的精
度就选 `l`。

这个家族不提供训练。`LibreMiDaS.train()` 无条件抛出 `NotImplementedError`。

## 验证

`val()` 运行的是共用的深度验证器：它用逐图像的最小二乘尺度和偏移，把每个预测对
齐到它的真值（ground truth），然后报告标准的零样本相对深度指标，AbsRel、RMSE
和三个 delta 阈值。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`，只是把检测框换成了 `depth_map`。

<code-tabs name="export" />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
