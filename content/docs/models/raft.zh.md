---
title: RAFT
families:
  - raft
seo_title: 在 Python 里用 RAFT 光流：帧间稠密运动，采用 BSD-3 许可
description: >-
  在 LibreYOLO 里运行 RAFT，对视频帧之间做稠密光流估计。提供 small 和 large 检查点以及 Sintel 和 KITTI
  微调版本，采用 BSD-3-Clause 许可。
lead: >-
  RAFT 是 Teed 和 Deng 提出的循环全对光流网络（ECCV 2020）。它接收两帧，为第一帧的每个像素返回它到第二帧的位移。LibreYOLO
  为光流任务移植了上游的推理代码，并镜像了四个已发布的检查点。
keywords:
  - RAFT
  - RAFT 光流
  - 光流 optical flow
  - 稠密光流 python
  - Recurrent All-Pairs Field Transforms
  - raft-things
  - raft-sintel
  - raft-kitti
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRAFTl-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRAFTl-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreRAFTl-flow.pt source=video.mp4 save=true
        arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRAFTl-flow.pt data=my_flow.yaml
source_hash: 5576df12c513723d
---

RAFT 在两帧的所有像素对之间构建一个 4D 相关体，并用一个循环更新模块细化同一个光流场。它是后来大多数光流网络的出发点，SEA-RAFT、FlowSeek 和 MEMFOF 都在其中。想要这个广为人知的基线时选它，素材接近 KITTI 或 Sintel 基准时选对应的微调版本。

## 安装

RAFT 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreRAFT` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。RAFT 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

| 尺寸 | 上游检查点 | 文件 |
| --- | --- | --- |
| `s` | raft-small | `LibreRAFTs-flow.pt` |
| `l` | raft-things（规范版本） | `LibreRAFTl-flow.pt` |
| `l` | raft-sintel 微调 | `LibreRAFTl-flow-sintel.pt` |
| `l` | raft-kitti 微调 | `LibreRAFTl-flow-kitti.pt` |

`LibreRAFT` 默认使用尺寸 `l`。推理在帧的原生分辨率上运行，像上游的 `InputPadder` 一样填充到 8 的倍数，细化迭代固定为 20 次（上游 `demo.py`）。`imgsz` 限制长边：更大的帧会被缩小，光流场再缩放回原始画布。

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

RAFT 是[普林斯顿视觉与学习实验室](https://github.com/princeton-vl/RAFT)（Princeton Vision & Learning Lab）的 Teed 和 Deng 的工作。上游推理代码按其 BSD-3-Clause 许可内置进来，上游下载脚本里的四个检查点（checkpoint）按同一许可镜像在 LibreYOLO 的 Hugging Face 组织上，不需要申请访问。BSD-3-Clause 允许商用和闭源使用，只要保留版权声明、条件和免责声明。

</provenance-box>

## 引用

<citation-block />
