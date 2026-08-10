---
title: YOLOv2
families:
  - yolo2
seo_title: 在 LibreYOLO 里运行 YOLOv2：预测、验证与导出
description: 在 LibreYOLO 里运行 YOLOv2（YOLO9000）：一个冻结的、仅推理的博物馆家族。可以预测、验证并导出，采用公共领域许可。
lead: >-
  YOLOv2 也以 YOLO9000 之名发表，是把锚框和 passthrough 层带进 YOLO 一脉的 Darknet-19
  检测器。LibreYOLO 以冻结、仅推理的展品形式收录它。
keywords:
  - YOLOv2
  - YOLO9000
  - Darknet
  - Darknet-19
  - yolov2 目标检测
  - 锚框
  - yolov2 权重下载
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO2b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## 安装

YOLOv2 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

这个家族只提供推理：`train()` 会抛出 `NotImplementedError`，所以本页没有训练一节。
预测、验证和导出都支持。权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改一行
的事。`conf` 过滤置信度阈值，`iou` 过滤 NMS 阈值，两者都作用在 `region` head 基于
锚框的预测上。数据源、流式处理和结果处理见[预测](/docs/predict)。

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
