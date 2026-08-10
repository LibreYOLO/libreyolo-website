---
title: FCOS
families: [fcos]
seo_title: "FCOS：在 LibreYOLO 里预测、验证和导出"
description: "在 LibreYOLO 里用 FCOS 做无锚框（anchor-free）目标检测。安装、预测、验证并导出这个采用 BSD-3-Clause 许可的 torchvision 移植版，ResNet-50/FPN。"
lead: "FCOS 逐像素检测目标，而不依赖一组预定义的锚框，它在特征图的每个位置上都预测一个检测框和一个中心度（centerness）分数。LibreYOLO 移植了 torchvision 的实现，用于检测。"
keywords: [FCOS, "fcos 目标检测", 无锚框检测, 单阶段检测器, torchvision, "fcos onnx 导出"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFCOSr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

FCOS 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行
改动。调用模型时不传任何阈值参数，会套用 FCOS 自己公布的默认值 `conf=0.2`、
`iou=0.6` 和 `max_det=100`；这三个里传进任意一个都会覆盖掉默认值。FCOS 在它逐
像素的预测之上保留了一个最终的 NMS 步骤。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

只有一种尺寸：带特征金字塔的 ResNet-50，这是这个家族认可的唯一变体。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

FCOS 可以导出到 ONNX、TorchScript 和 OpenVINO。FCOS 在计算图运行之前会保留原图
的宽高比，所以不管传进来的是什么，LibreYOLO 都会为 ONNX 和 OpenVINO 这两条路径
强制 `dynamic=True`，好让计算图对填充后的输入形状依然有效。导出的 `.onnx` 文件
按文件后缀经由 `LibreYOLO()` 重新加载，返回的还是同一个 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
