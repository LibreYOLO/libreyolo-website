---
title: DETR
families:
  - detr
seo_title: DETR：在 Apache-2.0 许可下预测和导出
description: >-
  在 LibreYOLO 里运行 DETR，最初的 detection transformer。安装、预测、验证并导出四种基于 ResNet 的尺寸，全部采用
  Apache-2.0 许可。
lead: >-
  DETR 是最初的 detection transformer，它用经过匈牙利匹配（Hungarian matching）的 transformer
  解码器预测一组固定数量的目标，而不是用锚框或稠密网格。LibreYOLO 提供四种尺寸，用于目标检测，只支持推理。
keywords:
  - DETR
  - detection transformer
  - 目标检测
  - 匈牙利匹配
  - transformer 解码器
  - detr 推理
  - detr onnx 导出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() 返回的是普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## 安装

DETR 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行
改动。`conf` 和 `max_det` 过滤查询（query）的选择；`iou` 为了 API 一致性会被接受，
但不起作用，因为解码器是一个集合预测器，没有 NMS 步骤。数据源、流式处理和结果
处理见[预测](/docs/predict)。

在 LibreYOLO 里 DETR 只支持推理。上游用匈牙利匹配训练 500 轮；那套配方这里没有
实现，所以 `train()` 会抛出 `NotImplementedError`。

## 变体

四个检查点（checkpoint）由两种骨干深度（ResNet-50 或 ResNet-101）和一个可选的
空洞 C5 阶段组合而成：DC5 变体把骨干的最后一个阶段保持在全分辨率，不再继续下
采样，所以解码器在相同的输入尺寸下读到更精细的特征图。四个变体都用 100 个学习
得到的目标查询和一个六层 transformer 编码器-解码器，输入分辨率也都相同。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何符合你训练所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。[导出](/docs/export)列出了每种格式
接受的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>
