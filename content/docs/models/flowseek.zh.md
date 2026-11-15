---
title: FlowSeek
families:
  - flowseek
seo_title: 在 Python 里用 FlowSeek 光流：借助深度基础模型先验
description: >-
  在 LibreYOLO 里运行 FlowSeek，用 Depth Anything V2 编码器和运动基做稠密光流估计。尺寸 t 和 s 采用
  Apache-2.0 许可；m 和 l 仅限非商用。
lead: >-
  FlowSeek 是 Poggi 和 Tosi 提出的光流网络（ICCV 2025），它在 SEA-RAFT 风格的循环网络上加入了一个冻结的 Depth
  Anything V2 编码器和由深度推导出的运动基。LibreYOLO 移植了它的四个尺寸；t 和 s 采用 Apache-2.0 许可，m 和 l
  仅限非商用。
keywords:
  - FlowSeek
  - 光流 optical flow
  - 稠密光流 python
  - Depth Anything V2
  - 运动基 motion bases
  - SEA-RAFT
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreFlowSeekt-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFlowSeekt-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreFlowSeekt-flow.pt source=video.mp4
        save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFlowSeekt-flow.pt data=my_flow.yaml
source_hash: b7af409bbc92d9c6
---

FlowSeek 从一个深度基础模型借来几何知识：Depth Anything V2 编码器的特征和一组由深度计算出的运动基会输入到循环光流更新中。托管的检查点（checkpoint）是上游最终的通用微调版本（TartanAir、Chairs、Things，然后是混合 Sintel、KITTI、HD1K 和 Things 的阶段），也就是上游在 Spring 和 LayeredFlow 上评估的那一个。商用项目选 `t` 或 `s`；`m` 和 `l` 带有一个非商用的编码器。

## 安装

FlowSeek 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreFlowSeek` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。FlowSeek 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

上游训练了两个网络；`s` 和 `l` 沿用 `t` 和 `m` 的权重，只是细化迭代次数更多，与上游的评估配置一致。

| 尺寸 | 上下文网络 | Depth Anything V2 编码器 | 迭代次数 | 权重许可证 |
| --- | --- | --- | ---: | --- |
| `t` | ResNet-18 | ViT-S | 4 | Apache-2.0 |
| `s` | ResNet-18 | ViT-S | 12 | Apache-2.0 |
| `m` | ResNet-34 | ViT-B | 4 | CC BY-NC 4.0，非商用 |
| `l` | ResNet-34 | ViT-B | 12 | CC BY-NC 4.0，非商用 |

文件名为 `LibreFlowSeek<size>-flow.pt`。`LibreFlowSeek` 默认使用尺寸 `m`，所以如果你需要可商用的权重，请显式传入 `t` 或 `s` 检查点。推理在原生分辨率上运行，在网络内部填充到 8 的倍数，填充后画布的每条边都必须至少 128 px。`imgsz` 限制长边。

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

FlowSeek 是[博洛尼亚大学](https://github.com/mattpoggi/flowseek)的 Poggi 和 Tosi 的工作。它的代码采用 Apache-2.0 许可，其中来自 SEA-RAFT 的部分采用 BSD-3-Clause 许可，已内置在 LibreYOLO 中。

检查点的许可证取决于尺寸，因为每个检查点都内嵌了一个 Depth Anything V2 编码器：

- `t` 和 `s` 内嵌 ViT-S 编码器，采用 **Apache-2.0** 许可：允许商用和闭源使用。
- `m` 和 `l` 内嵌 ViT-B 编码器，它采用 **CC BY-NC 4.0** 许可：**仅限非商用**，需要署名。它们的模型卡带有非商用提示横幅，LibreYOLO 在加载时会打印一条提示。

上游没有单独发布检查点许可证，所以仓库的 Apache-2.0 许可被当作默示授予来适用，并受编码器许可证的限制。这些权重镜像在 LibreYOLO 的 Hugging Face 组织上，不需要申请访问。

</provenance-box>

## 引用

<citation-block />
