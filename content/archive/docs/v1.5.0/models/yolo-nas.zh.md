---
title: YOLO-NAS
families:
  - yolonas
seo_title: YOLO-NAS：在 LibreYOLO 里预测、训练与导出
description: 在 LibreYOLO 里用 YOLO-NAS 做检测和姿态。Deci.AI 的权重是专有的、仅限非商业用途，LibreYOLO 一个都不发布。
lead: >-
  一个卷积检测器，它的骨干和 neck 出自 Deci.AI 的架构搜索，由量化感知的 RepVGG 模块搭成。它的权重属于
  Deci.AI，许可仅限非商业用途，LibreYOLO 一个都不发布。
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - 目标检测
  - 姿态估计
  - yolo-nas 训练自己的数据集
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 磁盘上还没有的名字会从 Deci 的 CDN 拉取；下载时会先打印 Deci
        # 的许可条款，拿走文件就等于接受它们
        model = LibreYOLO("LibreYOLONASs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 姿态
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # -pose 后缀会选中姿态 head 和它自己那一套权重
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: 从零开始训练
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # 完全不碰 Deci 的检查点：模型从随机权重开始，所以这次训练的
        # 产物只源自你自己的数据
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: 在 COCO 上验证
      language: bash
      code: |
        # 自带的 COCO yaml 里内嵌了一段下载脚本，所以除非数据集已经在
        # 本地，否则需要显式授权
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## 安装

YOLO-NAS 除了基础包之外不需要任何额外依赖。

```bash
pip install libreyolo
```

## 预测

磁盘上还没有的检查点（checkpoint）名字会从 Deci 的公共 CDN 拉取，而不是从
LibreYOLO 组织拉取，那里不托管这些权重中的任何一个。传输开始之前，库会每个进程
打印一次 Deci 的许可条款；下载下来的文件被打开之前，它的 SHA-256 会先和一个钉死
的值做校验。这些条款允许什么，见[许可证](#licensing)。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行的
改动。`conf` 设置置信度阈值，`iou` 设置 NMS 阈值。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

检测和姿态是同一套架构配上不同的 head，接受的参数也一样。下表里的尺寸是检测的
那一组；姿态在这些尺寸之外还多发布了一个更小的尺寸。姿态 head 预测的是 COCO 的
关键点集合。

<benchmark-table task="detect" />

<va-embed />

## 训练

<code-tabs name="train" />

不传 `epochs`、`lr0` 和 `amp` 时，它们会按任务分别解析，所以一次姿态训练的默认值
和一次检测训练不同。优化器默认是 AdamW。类别数来自数据集 YAML，head 会在第一轮
之前为它重建；在姿态 head 上，关键点数量也是同样的处理方式，所以一个 COCO 姿态
检查点可以微调到另一种规模的骨架上。

微调是从 Deci 的权重开始的，而这正是 Deci 的许可覆盖的部分。从随机初始化的模型
开始训练则完全不涉及任何 Deci 检查点，上面第三个代码片段就是这条路。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，衡量对象是任何符合你训练时所用格式的数据集。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。在不装 LibreYOLO 的裸运行时里跑
这张图也是支持的，但那时预处理和后处理就得你自己写。每种格式安装的 extra 不同，
各自也接受几个自己的参数。这两件事都在对应格式的页面上。

一次导出只是把同样的权重换一个容器再存一份。导出一个 Deci 检查点既不会改变权重
的来源，也不会改变覆盖它们的许可。

<code-tabs name="export" />

## 检查点

这里没有可列的。Deci 的许可禁止再分发，所以 LibreYOLO 组织不发布任何 YOLO-NAS
权重，下载会解析到别处：形如 `LibreYOLONAS<size>.pt` 的名字，或者姿态用的
`LibreYOLONAS<size>-pose.pt`，会映射到 Deci 公共 CDN 上对应的对象。

只有 SHA-256 被库钉死的检查点才能这样拉取。其他的一律失败关闭，而不是去打开一个
未经校验的第三方 pickle，得手动下载并以路径的形式传进来。已经在磁盘上的文件按
路径加载，没有下载，也没有校验和这道关。这也包括保持原始文件名的 Deci `.pth`，
加载器认得它。

## 许可证

<provenance-box>

LibreYOLO 既不托管也不镜像这些权重：LibreYOLO 的 Hugging Face 组织里没有任何属于
这个家族的东西。所有自动下载都改为走 Deci 的公共 CDN，开始之前按进程打印一次
Deci 的条款，并在文件被打开之前对照一个钉死的 SHA-256 做校验。

另一条路是从随机初始化的模型开始训练。这套架构在上游采用 Apache-2.0 许可，在这里
采用 MIT 许可，所以用你自己的数据这样训练出来的模型不派生自任何 Deci 检查点。

</provenance-box>

## 引用

YOLO-NAS 发布时没有论文。下面这一条是它的作者要求引用的，指向 SuperGradients，
也就是它随之发布的那个库。

<citation-block />
