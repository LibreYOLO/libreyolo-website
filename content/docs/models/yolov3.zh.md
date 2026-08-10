---
title: YOLOv3
families: [yolo3]
seo_title: "LibreYOLO 里的 YOLOv3：预测、验证与导出"
description: "在 LibreYOLO 里运行 YOLOv3：一个冻结的、仅推理的博物馆家族，提供 tiny、base 和 SPP 三种尺寸。可以预测、验证并导出，采用公有领域许可。"
lead: "YOLOv3 是那个基于 Darknet-53 的检测器，它给 YOLO 这条线加上了多尺度预测和相互独立的逻辑回归分类器。LibreYOLO 以冻结、仅推理的展品形式收录它的 tiny、base 和 SPP 三种尺寸。"
keywords: [YOLOv3, Darknet, Darknet-53, "yolov3 目标检测", "yolov3 pytorch", "yolov3 权重下载"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO3b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: SPP 尺寸
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # SPP 变体在检测 head 之前加了一个空间金字塔池化块，
        # 并以它自己的原生输入尺寸运行
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

YOLOv3 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

这个家族只提供推理：`train()` 会抛出 `NotImplementedError`，所以本页没有训练一节。
预测、验证和导出都支持。权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改一行
的事。`conf` 过滤置信度阈值，`iou` 过滤 NMS 阈值，两者都在三个 head 的检测框合并
之前逐尺度施加。数据源、流式处理和结果处理见[预测](/docs/predict)。

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
