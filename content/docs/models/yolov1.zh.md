---
title: YOLOv1
families:
  - yolo1
seo_title: 在 LibreYOLO 里运行 YOLOv1：预测、验证与导出
description: 在 LibreYOLO 里运行最初的 YOLOv1 检测器：一个冻结的、仅推理的博物馆家族。可以预测、验证并导出，采用公共领域许可。
lead: >-
  YOLOv1 是 2016 年那个最初的检测器，YOLO 家族的名字就来自它：一个卷积网络配上一个全连接
  head，一次前向就预测出所有检测框和类别分数，不用锚框。LibreYOLO 以冻结、仅推理的展品形式收录它。
keywords:
  - YOLOv1
  - yolov1 目标检测
  - Darknet
  - Pascal VOC
  - yolo 第一个版本
  - yolov1 预训练权重
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## 安装

YOLOv1 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

这个家族只提供推理：`train()` 会抛出 `NotImplementedError`，所以本页没有训练一节。
预测、验证和导出都支持。权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改一行
的事。有两点是这个家族特有的。已发布的检查点（checkpoint）是在 Pascal VOC
（2007+2012）上训练的，不是 COCO，所以 `box.cls` 索引的是 VOC 的 20 个类别
（aeroplane、bicycle、bird、boat、bottle、bus、car、cat、chair、cow、diningtable、
dog、horse、motorbike、person、pottedplant、sheep、sofa、train、tvmonitor），而不是
COCO 的 80 个。另外，全连接的检测 head 一次只接受一张图片，所以给它一个数据源列表
是循环跑完的，而不是作为真正的批次运行。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在一个使用检查点训练时那套 VOC 风格标签空间的数据集上测得。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。在不装 LibreYOLO 的裸运行时里跑这张
计算图也是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>
