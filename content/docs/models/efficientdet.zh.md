---
title: EfficientDet
families:
  - efficientdet
seo_title: EfficientDet：在 LibreYOLO 里做目标检测
description: >-
  在 LibreYOLO 里运行 EfficientDet D0-D4：采用 Apache-2.0 许可的 BiFPN 检测器，可以预测、验证并导出到
  ONNX、TensorRT 和 OpenVINO。
lead: >-
  EfficientDet 把 EfficientNet
  骨干和重复堆叠的双向特征金字塔网络（BiFPN）配在一起，并在五种尺寸上同时缩放深度、宽度和分辨率。LibreYOLO 以仅推理的检测器形式提供它。
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - efficientdet 目标检测
  - 复合缩放
  - efficientdet onnx 导出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## 安装

EfficientDet 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改
一行的事。EfficientDet 先解码基于锚框的候选，再做逐类别的非极大值抑制，所以
`conf`、`iou` 和 `max_det` 在这里都会真正起作用。数据源、流式处理和结果处理
见[预测](/docs/predict)。

## 变体

五种尺寸，从 D0 到 D4。每上一档都会把更大的 EfficientNet 骨干和更深、更宽的
BiFPN 以及更深的预测 head 配在一起，所以参数量和计算量一起增长，遵循论文的复合
缩放（compound scaling）规则。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物会按文件后缀通过 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点（checkpoint）一样，返回的也是同一个 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

LibreYOLO 的 D0-D4 检查点是通过采用 Apache-2.0 许可的 rwightman/efficientdet-pytorch
项目转换而来的，该项目本身镜像了 google/automl 里官方用 TensorFlow 训练出的权重，
没有改动学到的张量。采用 LGPL 许可的 zylo117/Yet-Another-EfficientDet-Pytorch
项目里的源码没有被参考或使用。

</provenance-box>
