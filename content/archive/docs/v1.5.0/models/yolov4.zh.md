---
title: YOLOv4
families:
  - yolo4
seo_title: YOLOv4：在 LibreYOLO 里运行、验证与导出
description: >-
  在 LibreYOLO 里运行 YOLOv4：一个冻结的、仅推理的博物馆家族，采用 CSPDarknet-53
  骨干。可以预测、验证并导出，采用公有领域许可。
lead: >-
  YOLOv4 把 CSPDarknet-53 骨干、一个 SPP 块和一个 PANet neck 与 Mish 激活组合在一起。LibreYOLO
  以冻结、仅推理的展品形式收录它的 tiny 和 base 两种尺寸。
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - yolov4 目标检测
  - yolov4 pytorch
  - mish 激活函数
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO4b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO4b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## 安装

YOLOv4 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

这个家族只提供推理：`train()` 会抛出 `NotImplementedError`，所以本页没有训练一节。
预测、验证和导出都支持。权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改一行
的事。`conf` 过滤置信度阈值，`iou` 过滤 NMS 阈值，两者都在每个 head 自己的
`scale_x_y` 中心缩放之后施加。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你验证时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点（checkpoint）一样，返回同样的 `Results`。在不装 LibreYOLO 的裸
运行时里跑这张计算图也是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
