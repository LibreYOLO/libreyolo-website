---
title: UniMatch
families:
  - unimatch
seo_title: 在 Python 里用 UniMatch GMFlow 光流：全局匹配，采用 MIT 许可
description: >-
  在 LibreYOLO 里运行 UniMatch（GMFlow），通过全局匹配做稠密光流估计。提供尺寸 s 和 l 以及 Sintel 和 KITTI
  微调版本，采用 MIT 许可。
lead: >-
  UniMatch 用一个 transformer 匹配器统一了光流、立体匹配和深度估计（TPAMI 2023）；GMFlow
  是其中的光流模型。LibreYOLO 用 GMFlow 检查点提供光流任务，采用 MIT 许可。
keywords:
  - UniMatch
  - GMFlow
  - GMFlow+
  - 光流 optical flow
  - 全局匹配 global matching
  - 稠密光流 python
  - flo 文件
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")   # 权重在首次使用时下载
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32，(dx, dy) 以像素为单位
        print(flow.mean_magnitude)    # 平均位移，单位为像素
    - label: 视频
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreUniMatchl-flow.pt")


        # N 帧给出 N - 1 个结果；第 i 个结果是从第 i 帧到第 i + 1 帧的光流

        for result in model("video.mp4", stream=True, save=True,
        save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreUniMatchl-flow.pt
        source=frame_0001.jpg,frame_0002.jpg save=true

        libreyolo predict model=LibreUniMatchl-flow.pt source=video.mp4
        save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均端点误差，单位为像素
        print(metrics["metrics/px1"])      # 误差小于 1 px 的像素占比
        print(metrics["metrics/fl_all"])   # KITTI 离群率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreUniMatchl-flow.pt data=my_flow.yaml
source_hash: 9a4b34da82ff287d
---

GMFlow 把光流当作一个全局匹配问题：一个 transformer 比较两帧的特征，对所有位置做 softmax 得到匹配结果，还可以再用局部回归细化。内置的上游代码也包含立体匹配和深度的路径，但 LibreYOLO 只接入了光流。像上游推荐的那样，处理真实世界素材时选 `l`。

## 安装

UniMatch 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。如果你更想直接构建模型，`libreyolo` 也导出了 `LibreUniMatch` 类。

<code-tabs name="predict" />

光流至少需要两帧：只传一张图像会报错。图像列表、按文件名排序的目录、视频或流都算作一个序列，`N` 帧给出 `N - 1` 个结果。`result.flow` 是一个 `FlowField`，带有一个位于第一帧原始画布上的 `(H, W, 2)` float32 数组；`flow.data[y, x]` 是以像素为单位的 `(dx, dy)`，`+x` 向右，`+y` 向下。UniMatch 不填充置信度图，所以 `flow.confidence` 为 `None`。`result.boxes` 为 `None`。`save=True` 写出 Middlebury 色轮图，`arrows=True` 或 `warp=True` 选择其他视图，`save_flow=True` 还会把每对帧的原始光流场写成 `.flo` 文件。[光流任务](/docs/tasks/optical-flow)页面完整介绍了载荷和各种视图。

## 变体

| 尺寸 | 上游检查点 | 文件 |
| --- | --- | --- |
| `s` | `gmflow-scale1-things`：1/8 尺度上的全局匹配（先 Chairs，后 Things） | `LibreUniMatchs-flow.pt` |
| `l` | `gmflow-scale2-regrefine6-mixdata`：两个尺度、6 步细化的 GMFlow+（规范版本） | `LibreUniMatchl-flow.pt` |
| `l` | Sintel 微调 | `LibreUniMatchl-flow-sintel.pt` |
| `l` | KITTI-2015 微调 | `LibreUniMatchl-flow-kitti.pt` |

只有 `l` 有 Sintel 和 KITTI 微调版本。`LibreUniMatch` 默认使用尺寸 `l`。预处理复现上游的 `inference_flow`：竖幅帧会被转置为横幅，两帧都会被缩放，使每条边向上取整到该尺寸的填充因子（`s` 为 16，`l` 为 32）。光流场再缩放回来，向量也相应重缩放。相对上游有一处有意的修正：对竖幅帧，上游在转置回来之后把 `dx` 和 `dy` 留在互换的状态；LibreYOLO 把它们换回来，让光流场符合光流约定。

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

UniMatch 是[自主视觉组](https://github.com/autonomousvision/unimatch)（Autonomous Vision Group，图宾根大学）的工作。上游 `unimatch` 包按其 MIT 许可原样内置进来。上游发布 GMFlow 检查点（checkpoint）时没有单独的权重许可证，所以 LibreYOLO 按发布项目的 MIT 许可托管它们：这是一种默示授予，是 LibreYOLO 的解读，未经发布者确认。不需要申请访问。MIT 允许商用和闭源使用，只要副本附带版权和许可声明。

</provenance-box>

## 引用

<citation-block />
