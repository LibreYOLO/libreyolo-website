---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: Depth Anything 3：在 LibreYOLO 里预测单目深度
description: >-
  在 LibreYOLO 里用 Depth Anything 3 做单目深度估计。安装、预测、验证并导出 DA3MONO-LARGE 检查点，采用
  Apache-2.0 许可。
lead: >-
  Depth Anything 3 就是一个普通的 DINOv2
  transformer，训练目标是从一个或多个视角预测深度和相机几何，没有任何架构上的特化。LibreYOLO 为深度任务移植了它的
  DA3MONO-LARGE 检查点：支持预测和零样本（zero-shot）验证，没有训练路径。
keywords:
  - Depth Anything 3
  - DA3
  - 单目深度估计
  - DINOv2
  - 相对深度
  - 深度图 python
  - 单张图片估计深度
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 读取深度图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap：稠密的 (H, W)，数值越大越近
        raw = depth.data                # 张量，没有度量单位，也没有跨图像的尺度
        normalized = depth.normalized() # 重新缩放到 [0, 1] 以便可视化
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## 安装

Depth Anything 3 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`result.depth_map` 装的是一张稠密的相对逆深度图：数值越大表示离相机越近，而且这些
数值没有度量单位，也没有跨图像的尺度。上游检查点（checkpoint）输出的是正的相对
深度；LibreYOLO 的网络包装器把它取逆，并复现了官方的天空处理，让输出符合
LibreYOLO 共享的深度契约。`save=True` 会把这张图经过色彩映射的可视化结果写入
磁盘；`Results.plot()` 不覆盖这个家族，因为它只为表面法线和边缘定义。数据源、
流式处理和结果处理见[预测](/docs/predict)。

## 变体

只有一种尺寸 `l`，输入分辨率固定。上游 DA3 还发布了 Small 和 Base 的 any-view
检查点、一个度量深度（metric depth）检查点，以及 Nested 和 Giant 检查点；
LibreYOLO 一个都没有对外提供。度量深度需要一套不同于 LibreYOLO 相对逆深度任务的
公开契约，而 any-view 和 Nested 检查点需要一套 LibreYOLO 并不提供的多图像相机
API。Large 和 Giant 的 any-view 检查点还采用 CC-BY-NC-4.0 许可，任何 LibreYOLO
下载路径都没有引用它们。

这个家族不提供训练。`LibreDepthAnything3.train()` 无条件抛出
`NotImplementedError`；请在上游训练，再用
`weights/convert_depth_anything3_weights.py` 转换一个兼容的 DA3MONO-LARGE
检查点。

## 验证

`val()` 运行共用的深度验证器：它用逐图像的最小二乘尺度和偏移把每个预测对齐到它的
真值（ground truth），然后报告标准的零样本相对深度指标，即 AbsRel、RMSE 和三个
delta 阈值。

<code-tabs name="val" />

## 导出

<export-matrix />

这个家族的导出限定在五种格式：ONNX、TorchScript、ExecuTorch、TensorRT 和
OpenVINO。请求其他任何格式都会抛出 `NotImplementedError`，而不是去尝试一次未经
验证的转换。导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或
`.engine` 文件的表现和检查点一样，返回同样的 `Results`，只是用 `depth_map` 取代
检测框。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
