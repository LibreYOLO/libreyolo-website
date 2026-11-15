---
title: 光流
seo_title: LibreYOLO 中的光流估计
description: 在 LibreYOLO 中用 RAFT、SEA-RAFT、WAFT、FlowSeek、DPFlow、UniMatch、NeuFlow v2 或 MEMFOF 估计视频帧之间的稠密光流，保存 .flo 文件并验证端点误差。
lead: 光流为一帧中的每个像素预测它在下一帧中移动到哪里。LibreYOLO 把它作为 flow 任务提供，1.8.0 新增：输入一对帧，输出一整片稠密的像素位移场，同一个调用背后有八个家族。
keywords:
  - 光流 python
  - raft 光流
  - sea-raft
  - 稠密运动估计
  - flo 文件
  - 光流 端点误差 epe
  - 视频运动场
last_verified: "1.8.0"
snippets:
  predict:
    - label: 两帧
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)           # (H, W, 2) float32，以像素为单位的 (dx, dy)
        print(flow.mean_magnitude)       # 平均位移，单位为像素
    - label: 视频或帧列表
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")

        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流
        for result in model("video.mp4", stream=True, vid_stride=2):
            print(result.flow.max_magnitude)
    - label: 可视化视图与原始场
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")
        frames = ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]

        model(frames, save=True)                    # 色轮（默认）
        model(frames, save=True, arrows=True)       # 在帧上画箭头
        model(frames, save=True, warp=True)         # 变形检查
        results = model(frames, save=True, save_flow=True)   # 同时写出 .flo

        results[0].flow.save("pair0.npy")           # .flo 或 .npy
  cli:
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSEARAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreSEARAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
        libreyolo val model=LibreSEARAFTm-flow.pt data=my_flow.yaml
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/px3"])
        print(metrics["metrics/fl_all"])   # KITTI 离群率
source_hash: ba3dde4eb92b3f1c
---

## 定义

`flow` 任务接收一个帧序列，为每一对相邻帧预测每个像素从第 `t` 帧到第 `t + 1` 帧的位移。
别名 `optical-flow`、`optical_flow` 和 `opticalflow` 都解析到它，检查点（checkpoint）带有
`-flow` 后缀。

一次预测会填充 `result.flow`，这是一个 `FlowField`，装着第一帧原图画布上的 `(H, W, 2)`
float32 数组。`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。`.dx`、`.dy`、
`.magnitude`、`.angle`、`.max_magnitude` 和 `.mean_magnitude` 用来读取它，`.warp(image)` 用这个场把一张图像
变形回去。`flow.confidence` 是一张可选的 `(H, W)` 图，取值在 `[0, 1]`，只有会预测它的家族才会填充
（SEA-RAFT，来自它的不确定性 head）；其他家族把它留作 `None`。`flow.next_frame` 保存第二帧，供变形视图使用。
`result.boxes` 为 `None`。

## 模型

八个家族支持 `flow`。它们都还没有模型页面；按检查点名称加载，见下面的代码片段。

| 家族 | 尺寸 | 权重许可证 |
|---|---|---|
| SEA-RAFT | s、m、l | BSD-3-Clause |
| RAFT | s、l，以及 Sintel 和 KITTI 变体 | BSD-3-Clause |
| NeuFlow v2 | s，以及 Sintel 和 Things 变体 | Apache-2.0 |
| WAFT | t、s、m、l | BSD-3-Clause；`t` 使用 DINOv3，带有 DINOv3 License |
| FlowSeek | t、s、m、l | t 和 s 采用 Apache-2.0；m 和 l 仅限非商用（CC BY-NC，Depth Anything V2 编码器） |
| DPFlow | m，以及 Sintel、KITTI 和 Spring 变体 | 仅限非商用（CC BY-NC-SA 4.0） |
| UniMatch | s、l，以及 Sintel 和 KITTI 变体 | MIT |
| MEMFOF | m，以及五个变体 | BSD-3-Clause |

检查点遵循 `Libre<Family><size>-flow.pt` 的命名，例如 `LibreSEARAFTm-flow.pt`、`LibreRAFTl-flow-kitti.pt`
或 `LibreNeuFlowv2s-flow.pt`。非商用权重在加载时会打印一条提示。NeuFlow v2 是小型的边缘模型。MEMFOF
每对帧读取三帧，运行器会在序列的开头和结尾重复边缘帧。这八个家族都记录了与上游代码的一致性对齐，权重都镜像在
LibreYOLO 的 Hugging Face 组织上。

## 预测

<code-tabs name="predict" />

单张图像会抛错：光流至少需要两帧。图像的列表或元组、按名称排序的目录、视频或流都算一个序列，`N` 帧给出 `N - 1`
个结果。第 `i` 个结果携带从第 `i` 帧到第 `i + 1` 帧的光流，以第 `i` 帧作为它的 `orig_img`。在视频上，
`vid_stride` 设定帧间隔，`stream=True` 每对帧产出一个结果。

有三种视图。默认是 Middlebury 色轮：色相表示方向，饱和度表示幅值，零运动是白色。`arrows=True` 在帧上每 16 个像素
画一个箭头。`warp=True` 把第 `t` 帧和变形回它上面的第 `t + 1` 帧并排显示，场正确时两个面板应当一致。
`save=True` 写出所选视图，`save_flow=True` 还会把每对帧的原始场写成 Middlebury `.flo` 文件，命名为
`<frame stem>_<frame index>.flo`。`FlowField.save(path)` 直接写出 `.flo` 或 `.npy`。

在 shell 里，把帧作为逗号分隔的 `source` 传入，并加上 `arrows=true`、`warp=true` 或 `save_flow=true`：

<code-tabs name="cli" />

输入源和流式处理见[预测](/docs/predict)，每一个 CLI flag 见 [`libreyolo predict`](/docs/cli/predict)。

## 数据集格式

光流数据集是一组帧序列。帧放在 `images/<split>/<sequence>/` 下，按文件名排序；同一个序列目录里的每一对相邻帧
就是一个样本，它的真值（ground truth）放在 `flows/` 下，相对路径相同，文件名取第一帧的主文件名。

```text
dataset/
  data.yaml
  images/
    val/alley_1/frame_0001.png
    val/alley_1/frame_0002.png
    val/alley_1/frame_0003.png
  flows/
    val/alley_1/frame_0001.flo
    val/alley_1/frame_0002.flo
```

```yaml
path: dataset
val: images/val
flows_dir: flows
flow_format: flo
```

`flow_format` 可以是 `flo`（Middlebury，默认）、`kitti`（16 位 RGB PNG，蓝色通道作为有效性掩码）或 `npy`
（一个 `(H, W, 2)` 浮点数组，其中非有限值视为无效）。每对帧可选的掩码 `<stem>_valid.png` 用非零值标记有效像素；
`flow_valid_suffix` 可以修改这个后缀。序列的最后一帧没有光流文件，不同目录中的帧永远不会配对。

## 训练

光流只提供推理。在光流家族上调用 `train()` 会抛出 `NotImplementedError`。

## 验证

`val()` 直接在有效像素上比较预测向量和真值向量，不做任何对齐。

<code-tabs name="val" />

`metrics/epe` 是平均端点误差，即预测向量与真实向量之间以像素为单位的欧氏距离，越低越好。`metrics/px1` 和
`metrics/px3` 是误差小于 1 和 3 个像素的有效像素比例。`metrics/fl_all` 是 KITTI 离群率：误差同时超过 3 个像素
和真实幅值 5% 的像素。最佳检查点的 fitness 是 `metrics/px1`，因为它随质量提高而增大。

## 导出

在某个家族实现双输入图和后端缩放契约之前，光流导出对所有格式都被阻止。见[导出矩阵](/docs/reference/export-matrix)。
