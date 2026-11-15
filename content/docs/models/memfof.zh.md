---
title: MEMFOF
families:
  - memfof
seo_title: 在 Python 里用 MEMFOF 光流：省显存的多帧光流
description: 在 LibreYOLO 里运行 MEMFOF，在 Full HD 视频上做省显存的三帧光流估计。一个网络、六个检查点，采用 BSD-3-Clause 许可。
lead: >-
  MEMFOF 是莫斯科国立大学图形与媒体实验室（MSU Graphics & Media Lab）提出的省显存多帧光流网络（ICCV 2025
  Highlight）。它读取 t-1、t 和 t+1 三帧，预测中间帧的光流，面向 Full HD 视频。它是 LibreYOLO
  里唯一读取三帧窗口的光流家族。
keywords:
  - MEMFOF
  - 多帧光流 multi-frame optical flow
  - 光流 optical flow
  - Full HD 光流
  - 视频光流
  - 稠密光流 python
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreMEMFOFm-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMEMFOFm-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreMEMFOFm-flow.pt source=video.mp4 save=true
        arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMEMFOFm-flow.pt data=my_flow.yaml
source_hash: 6df7e06e52ee9087
---

MEMFOF 是 RAFT 的后继，一次前向就预测中间帧的后向（`t` 到 `t-1`）和前向（`t` 到 `t+1`）光流。LibreYOLO 把前向光流作为 `Results.flow` 返回；后向光流会被计算，但目前不在任务约定之内。运行器会把 `(t-1, t, t+1)` 窗口交给它，并在序列开头把第 0 帧重复用作 `t-1`，所以你仍然只需传入普通的列表、文件夹或视频。处理高分辨率的真实世界视频时选它。

## 安装

MEMFOF 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreMEMFOF` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。MEMFOF 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

一种架构（ResNet-34 FPN），尺寸 `m`，共六个检查点（checkpoint）：

| 文件 | 上游检查点 |
| --- | --- |
| `LibreMEMFOFm-flow.pt` | MEMFOF-Tartan-T-TSKH，作者推荐用于真实世界视频（规范版本） |
| `LibreMEMFOFm-flow-tartan-t.pt` | 较早发布的一个训练阶段 |
| `LibreMEMFOFm-flow-tartan.pt` | 较早发布的一个训练阶段 |
| `LibreMEMFOFm-flow-kitti.pt` | KITTI 微调 |
| `LibreMEMFOFm-flow-sintel.pt` | Sintel 微调 |
| `LibreMEMFOFm-flow-spring.pt` | Spring 微调 |

帧在原生分辨率上运行，并按上游的复制填充补到 16 的倍数。长边超过 `imgsz`（默认 1920，即 MEMFOF 的 Full HD 目标）的帧会先被缩小，光流场再缩放回来。推理使用 fp32 和 8 次细化迭代，均为上游默认值。

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

MEMFOF 是 [MSU Graphics & Media Lab](https://github.com/msu-video-group/memfof) 的工作。上游推理代码按其 BSD-3-Clause 许可内置进来；其中的注意力模块来自 GMA，采用 WTFPL 许可。Hugging Face 上的上游检查点标注为 bsd-3-clause，但没有附带许可证文件，所以 LibreYOLO 按代码仓库的 BSD-3-Clause 许可镜像这六个转换后的检查点，不需要申请访问。BSD-3-Clause 允许商用和闭源使用，只要保留版权声明、条件和免责声明。

</provenance-box>

## 引用

<citation-block />
