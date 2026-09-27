---
title: PIDNet
families:
  - pidnet
seo_title: PIDNet：预测并导出 MIT 许可的实时分割模型
description: 在 LibreYOLO 里用 PIDNet 做实时语义分割。安装、预测、验证并导出采用 MIT 许可的 s/m/l 三个 Cityscapes 检查点。
lead: >-
  一个三分支语义分割网络，在受比例积分微分（proportional-integral-derivative）启发的设计上加了一条专门的边界分支，目标是实时推理。LibreYOLO
  只把它用于语义分割。
keywords:
  - PIDNet
  - Cityscapes
  - 实时语义分割 python
  - pidnet 预训练权重
  - 语义分割 边界
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) 类别 id
        print(mask.classes)      # 图像中出现的类别 id，已排序
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## 安装

PIDNet 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。这个家族必须带 `-sem`
文件名后缀。

<code-tabs name="predict" />

语义分割返回的是每个像素一个类别 id，而不是检测框，所以 `result.semantic_mask`
在 `.data` 上带一个 `(H, W)` 数组，在 `.classes` 上带图像中出现的类别 id 列表。
`conf`、`iou` 和 `max_det` 为了 API 一致而被接受，但不起作用：模型用 argmax 给
每个像素分配一个类别，没有置信度阈值，也没有 NMS 步骤。数据源、流式处理和结果
处理见[预测](/docs/predict)。

## 变体

三种尺寸，输入都固定为 1024 px。已发布的检查点（checkpoint）是官方 PIDNet
Cityscapes 权重的转换版本，19 个类别。

LibreYOLO 不训练 PIDNet：对这个家族调用 `train()` 会抛出
`NotImplementedError`，上面的[支持层级](/docs/models)把它标为仅推理。

## 验证

`val()` 返回 `metrics/mIoU` 和 `metrics/pixel_accuracy`，在任何采用你训练所用
格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。[导出](/docs/export)列出了每种
格式接受的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
