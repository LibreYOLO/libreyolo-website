---
title: ZipDepth
families: [zipdepth]
seo_title: "ZipDepth：LibreYOLO 里的轻量单目深度"
description: "在 LibreYOLO 里用 ZipDepth 做轻量的单目深度估计。安装、预测、验证并导出两个采用 MIT 许可的检查点。"
lead: "ZipDepth 是一个紧凑的可重参数化 CNN，从 Depth Anything V2 Large 蒸馏而来，预测稠密的相对逆深度图。LibreYOLO 支持它的 depth 任务：预测和零样本验证，没有训练这一步。"
keywords: [ZipDepth, 单目深度估计, "边缘设备 深度估计", 相对深度, "深度图 python", "轻量级深度模型"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreZipDepthb-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: NPU/边缘检查点
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 同样的编码器，换成不用 unfold 的上采样 head，面向缺少
        # gather/unfold 支持的编译器。输出在视觉上与 b 检查点等价
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## 安装

ZipDepth 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`result.depth_map` 带的是一张稠密的相对逆深度图：值越大表示离相机越近，而且这些
值没有度量单位，也没有跨图像的统一尺度。`save=True` 会把这张图经过色彩映射的可
视化结果写入磁盘；`Results.plot()` 不覆盖这个家族，因为它只为表面法线和边缘定
义。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

两个检查点（checkpoint），编码器容量相同，区别只在训练出来的上采样 head。`b`
用的是 convex 上采样，跑在 GPU 或 CPU 上。`bnpu` 换成了不用 unfold 的解码器，面
向缺少 gather/unfold 支持的 NPU 和边缘编译器；它的输出按文档说明在视觉上与 `b`
等价。导出目标是受限的运行时就选 `bnpu`，其他情况选 `b`。

两个检查点都是从 Depth Anything V2 Large 的伪标注蒸馏来的，所以这个家族是
LibreYOLO depth 任务里紧凑、面向边缘的那一档，和更大的 Depth Anything V2 编码器
并列。

这个家族不提供训练。`LibreZipDepth.train()` 无条件抛出 `NotImplementedError`：
上游的配方是在一个很大的图像集合上蒸馏伪标注，没法作为一次 LibreYOLO 训练复现。
请到上游 [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) 训练，再
用 `weights/convert_zipdepth_weights.py` 转换结果。

## 验证

`val()` 运行的是共用的深度验证器：它用逐图像的最小二乘尺度和偏移，把每个预测对
齐到它的真值（ground truth），然后报告标准的零样本相对深度指标，AbsRel、RMSE
和三个 delta 阈值。

<code-tabs name="val" />

## 导出

<export-matrix />

导出遵循固定分辨率的稠密约定：源图像被拉伸缩放到导出时的画布尺寸，返回的深度图
之后再缩放回原始画布。导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个
`.onnx` 或 `.ncnn` 文件的表现和检查点一样，返回同样的 `Results`，只是把检测框换
成了 `depth_map`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
