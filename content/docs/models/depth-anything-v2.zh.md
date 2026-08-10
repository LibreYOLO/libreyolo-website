---
title: Depth Anything V2
families:
  - depth_anything
seo_title: Depth Anything V2：预测并验证单目深度
description: >-
  在 LibreYOLO 里用 Depth Anything V2 做单目深度估计。安装、预测并验证；Small 采用 Apache-2.0 许可，Base
  和 Large 采用 CC-BY-NC-4.0。
lead: >-
  Depth Anything V2 是一个 DINOv2 编码器搭配 DPT 解码器，从单张图像预测稠密的相对逆深度图。LibreYOLO 支持它的
  depth 任务：预测和零样本验证，没有训练这一步。
keywords:
  - Depth Anything V2
  - 单目深度估计
  - DPT
  - DINOv2
  - 深度估计 python
  - 图像生成深度图
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 读取深度图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap：稠密的 (H, W)，值越大越近
        raw = depth.data                # 张量，没有度量单位，也没有跨图像的尺度
        normalized = depth.normalized() # 重新缩放到 [0, 1] 用于可视化
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## 安装

Depth Anything V2 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`result.depth_map` 带的是一张稠密的相对逆深度图：值越大表示离相机越近，而且这些
值没有度量单位，也没有跨图像的统一尺度。`save=True` 会把这张图经过色彩映射的可
视化结果写入磁盘；`Results.plot()` 不覆盖这个家族，因为它只为表面法线和边缘定
义。输入分辨率必须能被 14 整除，也就是 DPT head 所依赖的 DINOv2 patch 网格；
LibreYOLO 在运行前会检查这一点，不满足就抛出异常。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

四种编码器尺寸，s/b/l/g，对应 ViT-S/B/L/G。下面的检查点（checkpoint）表只列出
s、b 和 l；没有发布 Giant 检查点。四种共享同样的输入分辨率，所以选尺寸换的是编
码器容量，而不是图像尺寸。许可证也是一个因素：Small 检查点采用 Apache-2.0 许
可，而 Base 和 Large 采用 CC-BY-NC-4.0，见下面的许可证一节。

这个家族不提供训练和微调。`LibreDepthAnythingV2.train()` 无条件抛出
`NotImplementedError`；请改用 `weights/convert_depth_anything_v2_weights.py`
转换一个兼容的上游检查点。

## 验证

`val()` 运行的是共用的深度验证器：它用逐图像的最小二乘尺度和偏移，把每个预测对
齐到它的真值（ground truth），然后报告标准的零样本相对深度指标，AbsRel、RMSE
和三个 delta 阈值。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`，只是把检测框换成了 `depth_map`。
[导出](/docs/export)列出了每种格式接受的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
