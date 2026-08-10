---
title: Mask R-CNN
families: [mask_rcnn]
seo_title: "Mask R-CNN：在 LibreYOLO 里预测、验证和导出"
description: "在 LibreYOLO 里用 Mask R-CNN 做目标检测和实例分割。安装、预测、验证并导出这个采用 BSD-3-Clause 许可的 torchvision 移植版。"
lead: "Mask R-CNN 给 Faster R-CNN 加了一条按区域走的掩码分支，在它检测到的每个检测框旁边预测一张分割掩码。LibreYOLO 移植了 torchvision 的实现，用于目标检测和实例分割。"
keywords: [Mask R-CNN, 实例分割, "目标检测 python", "mask rcnn pytorch", Faster R-CNN, torchvision, "两阶段检测器"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMaskRCNNr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 仅检测框
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" 跳过掩码 head，从同一个检查点返回检测框，
        # 结果里不带掩码
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # 掩码
        print(metrics["metrics/mAP50-95(B)"])   # 检测框
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
---

## 安装

Mask R-CNN 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行
改动。不带 `task` 参数加载检查点返回的是实例掩码，因为分割是这个家族的默认任务；
`result.masks` 这时会把它们和检测框一起带出来。传 `task="detect"` 则加载同样的权重
但不带掩码 head，只返回检测框。`conf` 和 `iou` 设置置信度阈值和 NMS 阈值；和基于
查询的检测器不同，Mask R-CNN 保留了它上游的 NMS 步骤。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

只有一种骨干：ResNet-50 加一个特征金字塔，用的是 torchvision 的 v2 Mask R-CNN
builder。已发布的检查点采用 BSD-3-Clause 许可，同时服务于这个家族的两个任务，所以
没有尺寸需要在其中挑选。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典。对这个检查点默认的分割任务来说，普通的
`metrics/mAP50-95` 键里放的是掩码分数，同一次运行还会在 `(B)` 后缀下报告检测框，
所以一次跑完两者都有。

<code-tabs name="val" />

## 导出

<export-matrix />

Mask R-CNN 只能导出到 ONNX，批大小为 1。导出的计算图把上游的缩放和掩码粘贴步骤
保留在内部，所以无论传入什么，LibreYOLO 都会强制 `dynamic=True`，好让计算图对非
正方形的数据源依然有效。导出的 `.onnx` 文件按文件后缀经由 `LibreYOLO()` 重新加载，
返回同样的 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。下面这个唯一的检查点列在 detect 下面，但同一个文件
也能用于分割：不传 `task` 参数，它默认就返回掩码。

<checkpoint-table />

## 许可证

<provenance-box>

Mask R-CNN 是作为 LibreYOLO 的 Faster R-CNN 包装器的子类构建的：它共用同样的
torchvision 源码和 BSD-3-Clause 许可，并从同一个移植过来的 commit 里加上了掩码
预测器和掩码 RoI head。

</provenance-box>
