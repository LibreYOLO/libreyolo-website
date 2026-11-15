---
title: SEA-RAFT
families:
  - searaft
seo_title: 在 Python 里用 SEA-RAFT 光流：带置信度图的快速 RAFT，采用 BSD-3 许可
description: >-
  在 LibreYOLO 里运行 SEA-RAFT，对帧间做稠密光流估计，并由其不确定性 head 给出逐像素置信度图。提供尺寸 s、m 和 l，采用
  BSD-3-Clause 许可。
lead: >-
  SEA-RAFT（Simple, Efficient, Accurate RAFT）是 Wang、Lipson 和 Deng 提出的 RAFT
  后继（ECCV 2024）。它预测两帧之间的稠密光流，而且与 LibreYOLO 里的其他光流家族不同，还会给出逐像素置信度图。LibreYOLO
  移植了上游的推理代码，并镜像了它的 S 和 M 检查点。
keywords:
  - SEA-RAFT
  - SEARAFT
  - 光流 optical flow
  - 稠密光流 python
  - RAFT
  - 光流置信度
  - 光流不确定性 uncertainty
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSEARAFTm-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSEARAFTm-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreSEARAFTm-flow.pt source=video.mp4 save=true
        arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSEARAFTm-flow.pt data=my_flow.yaml
source_hash: 836f41517d948354
---

SEA-RAFT 保留了 RAFT 的全对相关金字塔，直接回归一个初始光流，用 ConvNeXt 更新模块细化它，并用混合拉普拉斯损失函数训练。LibreYOLO 正是把预测出的混合方差转成 `flow.confidence`。[光流任务页面](/docs/tasks/optical-flow)的示例用的就是这个家族；想在光流旁边得到逐像素置信度时选它。

## 安装

SEA-RAFT 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreSEARAFT` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。`flow.confidence` 是一张取值在 `[0, 1]` 内的 `(H, W)` 图，由预测出的混合方差推导而来：值越高，表示网络对该像素处的向量越有把握。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

| 尺寸 | 编码器 | 细化迭代次数 | 文件 |
| --- | --- | ---: | --- |
| `s` | ResNet18 | 4 | `LibreSEARAFTs-flow.pt` |
| `m` | ResNet34 | 4 | `LibreSEARAFTm-flow.pt` |
| `l` | ResNet34，即 `m` 的权重 | 12 | `LibreSEARAFTl-flow.pt` |

上游没有单独提供 L 权重：`l` 就是用 12 次而不是 4 次迭代运行的 `m` 检查点（checkpoint）。三者都是上游的 Tartan-C-T-TSKH-spring540x960 检查点。`LibreSEARAFT` 默认使用尺寸 `m`。长边不超过 960 px（即训练裁剪宽度）的帧在原生分辨率上运行，填充到 8 的倍数；更大的帧会被缩小，光流场再缩放回来。`imgsz` 可以改变这个上限。

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

SEA-RAFT 是[普林斯顿视觉与学习实验室](https://github.com/princeton-vl/SEA-RAFT)（Princeton Vision & Learning Lab）的 Wang、Lipson 和 Deng 的工作。上游代码中用于推理的子集按其 BSD-3-Clause 许可内置进来。上游发布 S 和 M 权重时没有单独的权重许可证；LibreYOLO 按仓库的 BSD-3-Clause 许可镜像它们，不需要申请访问。BSD-3-Clause 允许商用和闭源使用，只要保留版权声明、条件和免责声明。

</provenance-box>

## 引用

<citation-block />
