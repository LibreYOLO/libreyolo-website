---
title: NeuFlow v2
families:
  - neuflowv2
seo_title: 在 Python 里用 NeuFlow v2 光流：900 万参数的边缘模型，采用 Apache-2.0 许可
description: >-
  在 LibreYOLO 里运行 NeuFlow v2，在边缘设备上做快速的稠密光流估计。一个 900 万参数的网络，提供 mixed、Sintel 和
  Things 检查点，采用 Apache-2.0 许可。
lead: >-
  NeuFlow v2 是美国东北大学野外机器人实验室（Field Robotics Lab）为边缘设备打造的高效光流网络。它约有 900
  万参数，在固定画布上运行；它是 LibreYOLO 光流家族中的边缘模型。
keywords:
  - NeuFlow v2
  - NeuFlow
  - 光流 optical flow
  - 边缘设备 光流
  - 实时光流
  - 稠密光流 python
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNeuFlowv2s-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreNeuFlowv2s-flow.pt source=video.mp4
        save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNeuFlowv2s-flow.pt data=my_flow.yaml
source_hash: 22669af5be77f2b3
---

NeuFlow v2 把一个浅层 CNN 编码器与两个交叉注意力层以及 1/16 尺度上的全局 softmax 匹配组合起来，之后是 1/16 和 1/8 尺度上的轻量卷积细化器，以及一个学习得到的 8 倍 convex 上采样器。当速度和内存比最后一点精度更重要时选它，比如按帧率处理视频，或者在小型硬件上运行。

## 安装

NeuFlow v2 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreNeuFlowv2` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。NeuFlow v2 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

一种架构，尺寸 `s`，同一个网络有三种训练数据变体：

| 文件 | 上游检查点 |
| --- | --- |
| `LibreNeuFlowv2s-flow.pt` | `neuflow_mixed.pth`，在多个数据集上训练（规范版本） |
| `LibreNeuFlowv2s-flow-sintel.pt` | `neuflow_sintel.pth` |
| `LibreNeuFlowv2s-flow-things.pt` | `neuflow_things.pth` |

与其他光流家族不同，NeuFlow v2 不在原生分辨率上运行。两帧都会被拉伸缩放到一个固定画布，默认与上游 `infer.py` 一样为 432x768（高 x 宽）；`imgsz` 接受一个整数或一个 `(height, width)` 对，每条边都是 16 的倍数。网络在 1/16 上运行一步细化，在 1/8 上运行八步，光流场再缩放回原始画布，每个向量分量乘以对应轴的缩放比例。

不提供训练：`train()` 会抛出 `NotImplementedError`。

## 验证

`val()` 在光流数据集上运行共用的光流验证器（每个序列一个文件夹存放帧，真值（ground truth）为 Middlebury `.flo`、KITTI 16 位 PNG 或 `.npy`；见[数据集格式](/docs/tasks/optical-flow#dataset-format)）。它报告平均端点误差 `metrics/epe`、`metrics/px1` 和 `metrics/px3`（误差在 1 px 和 3 px 以内的有效像素占比），以及 KITTI 离群率 `metrics/fl_all`。

<code-tabs name="val" />

## 导出

<export-matrix />

目前还不支持导出任何格式。在某个家族实现光流任务的双输入计算图（`image1`、`image2`）和后端向量重缩放约定之前，光流导出都无法进行。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

NeuFlow v2 是[美国东北大学野外机器人实验室](https://github.com/neufieldrobotics/NeuFlow_v2)的工作。上游模型代码按其 Apache-2.0 许可内置进来。上游把它的三个检查点（checkpoint）提交在同一个 Apache-2.0 仓库里，没有单独的权重许可证，所以 LibreYOLO 按 Apache-2.0 镜像它们，不需要申请访问。Apache-2.0 允许商用和闭源使用；再分发时请保留许可证文本和署名声明。

</provenance-box>

## 引用

<citation-block />
