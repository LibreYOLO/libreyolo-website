---
title: DPFlow
families:
  - dpflow
seo_title: 在 Python 里用 DPFlow 光流：最高支持 8K 的双金字塔光流
description: 在 LibreYOLO 里运行 DPFlow，对高分辨率帧做稠密光流估计，金字塔会随输入尺寸自适应。代码采用 Apache-2.0 许可；权重仅限非商用。
lead: >-
  DPFlow 是 Henrique Morimitsu 等人提出的自适应光流估计器（CVPR
  2025）。它的双金字塔编码器会根据输入尺寸调整深度，所以在低分辨率上训练的模型可以泛化到 4K 和 8K 帧。LibreYOLO 从 ptlflow
  移植了它；权重采用 CC BY-NC-SA 4.0 许可，仅限非商用。
keywords:
  - DPFlow
  - 光流 optical flow
  - 高分辨率光流
  - 4K 光流
  - 双金字塔 dual pyramid
  - ptlflow
  - 稠密光流 python
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDPFlowm-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDPFlowm-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreDPFlowm-flow.pt source=video.mp4 save=true
        arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDPFlowm-flow.pt data=my_flow.yaml
source_hash: 6bdbd3161d29d017
---

DPFlow 根据图像对角线选择金字塔层数：1K 左右用三层，分辨率每翻一倍多加一层。LibreYOLO 保留了这条规则，并默认在帧的原生分辨率上运行，自适应深度正是在这里起作用。做高分辨率素材的研究时选它；它的权重仅限非商用。

## 安装

DPFlow 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreDPFlow` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。DPFlow 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

一种架构，尺寸 `m`，共四个检查点（checkpoint）：

| 文件 | 训练阶段 |
| --- | --- |
| `LibreDPFlowm-flow.pt` | FlyingThings3D（`things`），用于泛化的检查点（规范版本） |
| `LibreDPFlowm-flow-sintel.pt` | Sintel 微调 |
| `LibreDPFlowm-flow-kitti.pt` | KITTI 微调 |
| `LibreDPFlowm-flow-spring.pt` | Spring 微调 |

迭代计划固定为上游的测试设置，每个金字塔层级 4 次细化迭代。每对帧都独立估计：不使用上游跨帧的热启动。`imgsz` 限制长边（默认 8192，实际上等于不限制）；超过它的帧会被缩小，光流场再缩放回原始画布。

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

DPFlow 是 Henrique Morimitsu 等人的工作，发布在 [ptlflow](https://github.com/hmorimitsu/ptlflow) 中。代码和权重的许可证不同。

- 代码采用 **Apache-2.0** 许可：ptlflow 的 `models/dpflow` 连同源自 timm 的辅助代码（Apache-2.0）和改编自 RAFT 的输入填充器（BSD-3-Clause）一起被内置进来。
- 权重采用 **CC BY-NC-SA 4.0** 许可：**仅限非商用**，需要署名，而且你基于它们构建并分享的任何东西都必须采用相同的许可。DPFlow 的 README 还补充说，由于训练数据集的许可，这些权重仅用于学术和研究目的。请把全部四个检查点都当作研究用权重。

LibreYOLO 镜像这些权重时附上了许可证文本和非商用提示横幅，不需要申请访问，并在加载时打印一条提示。

</provenance-box>

## 引用

<citation-block />
