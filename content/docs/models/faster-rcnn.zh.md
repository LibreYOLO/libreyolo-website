---
title: Faster R-CNN
families: [faster_rcnn]
seo_title: "Faster R-CNN：在 LibreYOLO 里预测、验证和导出"
description: "在 LibreYOLO 里用四种骨干运行 Faster R-CNN 做目标检测。安装、预测、验证并导出这个采用 BSD-3-Clause 许可的 torchvision 移植版。"
lead: "Faster R-CNN 用一个区域建议网络（region proposal network）喂给两阶段分类器来检测目标，正是这个架构让区域建议成为同一个训练网络的一部分，而不再是单独的一步。LibreYOLO 移植了 torchvision 的实现，用于目标检测。"
keywords: [Faster R-CNN, 目标检测, "faster rcnn 目标检测", "区域建议网络", "两阶段检测器", torchvision]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFasterRCNNl.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

Faster R-CNN 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行
改动。`conf` 和 `iou` 设置置信度阈值和 NMS 阈值；和基于查询的检测器不同，
Faster R-CNN 保留了它上游的 NMS 步骤。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

四种尺寸，每一种都是不同的 torchvision 配置，而不是同一个配置按比例缩放的版本：
`n` 是 MobileNetV3-Large 配 320 px 输入，`s` 是同样的骨干配 800 px，`m` 是
ResNet-50 加一个特征金字塔，`l` 是 v2 修订版，用更深的区域建议 head 和一个四层
卷积的检测框 head 取代了 `m` 的那一套。`n` 和 `s` 用精度换来更轻的骨干。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何符合你训练所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

Faster R-CNN 只能导出到 ONNX，批大小为 1。导出的计算图把上游的缩放步骤保留在
内部，所以无论传入什么，LibreYOLO 都会强制 `dynamic=True`，好让计算图对非正方形
的数据源依然有效。导出的 `.onnx` 文件按文件后缀经由 `LibreYOLO()` 重新加载，返回
同样的 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
