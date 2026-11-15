---
title: WAFT
families:
  - waft
seo_title: 在 Python 里用 WAFT 光流：只靠 warp 的场变换，采用 BSD-3 许可
description: >-
  在 LibreYOLO 里运行 WAFT，不用代价体（cost volume）做稠密光流估计。按编码器分为 t、s、m、l 四个尺寸；s、m 和 l 采用
  BSD-3-Clause 许可，t 适用 DINOv3 License。
lead: >-
  WAFT（Warping-Alone Field Transforms）是 Wang 和 Deng 提出的光流方法（ICLR 2026）。它用高分辨率特征
  warp 和一个 ViT 细化循环取代了 RAFT 的代价体。LibreYOLO 移植了上游代码，并镜像了它的四个零样本检查点，每个特征编码器一个。
keywords:
  - WAFT
  - Warping-Alone Field Transforms
  - 光流 optical flow
  - 稠密光流 python
  - DINOv3
  - Depth Anything V2
  - Twins-SVT
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreWAFTm-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreWAFTm-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreWAFTm-flow.pt source=video.mp4 save=true
        arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreWAFTm-flow.pt data=my_flow.yaml
source_hash: 0cd50470188b142d
---

WAFT 去掉了 RAFT 及其后继构建的相关体，改为在每一步细化之前，用当前的光流估计对第二帧的高分辨率特征做 warp。全部四个检查点（checkpoint）都是上游的零样本模型（先 TartanAir，再 Chairs，再 Things）。除非你有理由选择某个特定编码器，否则选 `m`，也就是上游 README 推荐用于下游任务的 a1 检查点。

## 安装

WAFT 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreWAFT` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。WAFT 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

| 尺寸 | 上游检查点 | 参数量 | 文件 |
| --- | --- | ---: | --- |
| `t` | a2，DINOv3 ViT-S/16 编码器 | 56.5 M | `LibreWAFTt-flow-dinov3.pt` |
| `s` | a2，Depth Anything V2 ViT-S | 56.9 M | `LibreWAFTs-flow-dav2.pt` |
| `m` | a1，Depth Anything V2 ViT-S | 64.2 M | `LibreWAFTm-flow.pt` |
| `l` | a2，Twins-SVT-Large 编码器 | 136.0 M | `LibreWAFTl-flow-twins.pt` |

编码器就是文件名里的权重变体，每个尺寸只以表中的名字存在；任何其他组合都会报错，并给出已发布的名字。`LibreWAFT` 默认使用尺寸 `m`。预测在原生分辨率上运行：网络根据编码器把这对帧填充到 112、32 或 16 的倍数，再把光流裁剪回来。细化迭代固定为五次，即上游的评估默认值。`imgsz` 限制长边（默认 8192，实际上等于不限制）。

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

WAFT 是[普林斯顿视觉与学习实验室](https://github.com/princeton-vl/WAFT)（Princeton Vision & Learning Lab）的 Wang 和 Deng 的工作。上游代码按其 BSD-3-Clause 许可内置进来。上游没有发布单独的权重许可证，所以 LibreYOLO 按仓库的 BSD-3-Clause 许可，加上每个检查点内嵌编码器的许可证来镜像这些检查点，不需要申请访问。

- `s`、`m` 和 `l` 采用 BSD-3-Clause 许可。它们的编码器采用 Apache-2.0 许可（`s` 和 `m` 是 Depth Anything V2 Small，`l` 是源自 timm 的一份 Twins-SVT-Large 副本）。允许商用和闭源使用。
- `t` 包含一个由 WAFT 作者微调的 DINOv3 ViT-S/16，所以它适用 Meta 的 **DINOv3 License**：一份未经 OSI 批准的自定义许可证，有自己的使用限制和贸易管制条款。在商用 `t` 之前，请先读一遍。

</provenance-box>

## 引用

<citation-block />
