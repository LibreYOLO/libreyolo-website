---
title: CenterNet
families:
  - centernet
seo_title: CenterNet：在 LibreYOLO 里做目标检测
description: >-
  在 LibreYOLO 里用 ResDCN-18 和 DLA-34 骨干运行 CenterNet（Objects as Points）。采用 MIT
  许可，可以预测、验证并导出到 ONNX。没有训练路径。
lead: >-
  CenterNet
  把一个目标建模为它检测框的中心点，其余所有属性都从热力图的一个峰值回归得到，所以它既不需要锚框，也不需要非极大值抑制（non-maximum-suppression）步骤。LibreYOLO
  以仅推理的检测器形式提供它。
keywords:
  - CenterNet
  - Objects as Points
  - centernet 目标检测
  - 无锚框检测器
  - 关键点检测
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # 导出 ONNX 需要 opset 16 或更高：可变形卷积的上采样阶段会
        # 下降为 GridSample，而 GridSample 是 opset 16 引入的
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## 安装

CenterNet 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改
一行的事。`conf` 和 `max_det` 会过滤排好序的热力图峰值；`iou` 为了 API 一致性会
被接受，但不起作用，因为 CenterNet 的 top-k 峰值解码不需要基于检测框 IoU 的抑制
步骤。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

两种骨干。`resdcn18` 把 ResNet-18 主干和可变形卷积上采样配在一起；`dla34` 把
DLA-34 主干和迭代式深度聚合上采样配在一起。两者接的是同样的三个稠密 head（热力
图、宽/高、偏移），输入画布也一样。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出 ONNX 需要 opset 16 或更高：两种骨干里的可变形卷积上采样阶段都会下降为 ONNX
的 `GridSample` 算子，而这个算子是 opset 16 引入的。请求更低的 opset 会在追踪开始
前就抛出异常。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

ResDCN-18 的计算图还要归功于微软采用 MIT 许可的 human-pose-estimation.pytorch，
DLA-34 的计算图则归功于 Fisher Yu 采用 BSD-3-Clause 许可的 DLA 实现。LibreYOLO
没有随库内置上游项目当年使用的原版 DCNv2 扩展；原生执行改用 torchvision 采用
BSD-3-Clause 许可的 `deform_conv2d`，而仅用于导出的可移植实现是为 LibreYOLO 单独
编写的。

</provenance-box>

## 引用

<citation-block />
