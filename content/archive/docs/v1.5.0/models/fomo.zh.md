---
title: FOMO
families:
  - fomo
seo_title: FOMO：在 LibreYOLO 里做点定位、训练和导出
description: >-
  在 LibreYOLO 里运行 FOMO（Faster Objects, More
  Objects）：一个用来清点大量小目标的极小点定位检测器。安装、预测、训练和导出。
lead: FOMO 是一个基于网格的点定位器：低分辨率网格的每个格子被分类为背景或目标中心，完全不做检测框回归。LibreYOLO 在 point 任务下支持它。
keywords:
  - FOMO
  - Faster Objects More Objects
  - fomo 点定位
  - 中心点检测
  - 小目标检测
  - 边缘设备 目标检测
  - 单片机 目标检测
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMO 权重不会自动下载（见下面的检查点一节）
        # 把它指向一个你已经下载到本地的检查点
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz 必须显式传入：CLI 会把它默认成 640，而 s 检查点

        # 只接受它原生的 96

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## 安装

FOMO 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

和本站其他所有家族都不一样，LibreFOMO 的权重不会自动下载：
`LibreYOLO("LibreFOMOs-point.pt")` 会在磁盘上找这个文件，找不到就抛出一个点名该
文件的 `ValueError`，而不会去 Hugging Face 拉。请先从
[LibreYOLO 组织](https://huggingface.co/LibreYOLO)下载一个检查点（checkpoint），
再按本地路径加载，或者自己训练一个（见下面的训练一节）。

<code-tabs name="predict" />

结果携带的是 `points` 而不是 `boxes`：每一行是 `x, y, class, confidence`，可以通过
`result.points.data` 拿到，也可以通过 `.xy`、`.xyn`、`.cls` 和 `.conf` 这几个访问器
拿到。没有 `iou` 阈值可设，因为没有检测框需要抑制；`predict(..., nms_radius=1)`
控制两个检测结果之间必须相隔多少个网格单元才能同时留下来，而文件名必须带上 FOMO
的 `-point` 任务后缀，加载器才认得出来。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

三种尺寸 `s`、`m` 和 `l`，用的是逐级加宽的 MobileNetV2 风格骨干，输入分辨率也相应
变大并且固定，各自后面只接一个 1x1 的分类 head。这个家族在这里没有基准测试表；下面
表格里的检查点文件大小是目前已公布的、最能区分各个尺寸的信号。

## 训练

<code-tabs name="train" />

`imgsz` 不是随便选的：它默认取所加载检查点的原生分辨率，传别的值会抛出 `ValueError`
并点名它期望的尺寸。这些尺寸是 `s` 为 96、`m` 为 192、`l` 为 224。CLI 把 `imgsz`
默认成 640，所以 `libreyolo train` 命令必须显式设置它来匹配检查点。

其余不动的话，训练器会以批大小 32 跑 40 轮，用 Adam，`lr0=3e-4`，不做权重衰减，并在
逐格的交叉熵损失函数里把前景类相对背景加权 100 倍，因为典型场景里几乎每个网格单元都
是背景。EMA 和混合精度默认都是关的，LibreYOLO 别处用的几何和颜色数据增强一个也不用：
mosaic、mixup、HSV 抖动、翻转、旋转、平移和错切全都是零。

已发布的 LibreFOMO 检查点就是按这条路径训练出来的，在 COCO 上从零开始。

数据集和日志记录器见[训练](/docs/train)。

## 验证

`val()` 会分发到一个专为这个家族写的网格级验证器。除了和其他 point 任务共用的、基于
点匹配的 `metrics/precision`、`metrics/recall` 和 `metrics/mAP@` 这几个键之外，它还会
扫描置信度阈值和 `nms_radius` 的取值，把 F1 最好的那个组合发布在 `metrics/grid_F1`、
`metrics/grid_precision`、`metrics/grid_recall` 和 `metrics/grid_mean_distance` 下面，
并把产生这个组合的阈值和半径发布在 `decode/threshold` 和 `decode/nms_radius` 下面。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物会按文件后缀通过 `LibreYOLO()` 加载回来，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点一样，返回的也是同一个 `Results`。在裸运行时里跑这张计算图、完全
不装 LibreYOLO，同样是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。它们都不会自动下载：从链接的 Hugging Face 页面取下你
要的文件，再把它的本地路径传给 `LibreYOLO()`。

<checkpoint-table />

## 许可证

<provenance-box>

FOMO 没有可以链接的上游代码仓库：Edge Impulse 通过一篇博客文章和它的产品文档描述了
这项技术，但没有公开 FOMO 的训练或推理代码。这里的架构和训练是 LibreYOLO 依据那份
公开描述自己实现的，已发布的 LibreFOMO 检查点也是在 COCO 上从零训练的，所以代码和
这些权重都采用 MIT 许可，是 LibreYOLO 自己的。FOMO 这个名字以及它所描述的技术仍然
属于 Edge Impulse。

</provenance-box>
