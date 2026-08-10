---
title: RetinaNet
families: [retinanet]
seo_title: "RetinaNet：在 LibreYOLO 里预测、验证和导出"
description: "在 LibreYOLO 里用 RetinaNet 做单阶段目标检测，配 focal 损失。安装、预测、验证并导出这个采用 BSD-3-Clause 许可的 torchvision 移植版。"
lead: "RetinaNet 是一个用 focal 损失训练的单阶段检测器，focal 损失会压低简单负样本的权重，让一整片密集的锚框网格不再需要单独的候选框阶段也能保持精度。LibreYOLO 移植了 torchvision 的实现，用于检测。"
keywords: [RetinaNet, "retinanet 目标检测", "focal loss", 单阶段检测器, torchvision, "retinanet onnx 导出"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRetinaNetr50v2.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

RetinaNet 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行
改动。`conf` 和 `iou` 设置置信度阈值和 NMS 阈值；RetinaNet 在它密集的锚框网格之
上保留了上游的 NMS 步骤。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

两种尺寸，都是 ResNet-50 配特征金字塔：`r50` 是原始的 head，`r50v2` 把它换成一
个 GroupNorm head，外加一个更宽的 P6 块，这个块的输入取自骨干的最后一个阶段，而
不是 FPN 的输出。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

RetinaNet 只能导出到 ONNX，批大小为 1。RetinaNet 会把输入缩放到一个可变的、保持
宽高比的尺寸，所以无论传入什么，LibreYOLO 都会强制 `dynamic=True`，好让计算图对
不同形状的数据源依然有效。导出的 `.onnx` 文件按文件后缀经由 `LibreYOLO()` 重新
加载，返回同样的 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>
