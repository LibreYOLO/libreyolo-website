---
title: FCN
families: [fcn]
seo_title: "FCN：预测并导出 BSD-3-Clause 许可的 ResNet FCN"
description: "在 LibreYOLO 里用 FCN 做语义分割。安装、预测、验证并导出 torchvision 的膨胀 ResNet FCN 检查点。"
lead: "一个逐像素的稠密分类器，把检测器的全连接层换成卷积，于是输出的是全分辨率的类别图，而不是检测框。LibreYOLO 只把它用于语义分割。"
keywords: [FCN, 全卷积网络, "语义分割 python", "fcn 语义分割 预训练权重", ResNet]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) 类别 id
        print(mask.classes)      # 图像中出现的类别 id，已排序
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFCNr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## 安装

FCN 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

语义分割返回的是每个像素一个类别 id，而不是检测框，所以 `result.semantic_mask`
在 `.data` 上带一个 `(H, W)` 数组，在 `.classes` 上带图像中出现的类别 id 列表。
`conf`、`iou` 和 `max_det` 为了 API 一致而被接受，但不起作用：模型用 argmax 给
每个像素分配一个类别，没有置信度阈值，也没有 NMS 步骤。数据源、流式处理和结果
处理见[预测](/docs/predict)。

## 变体

两种 ResNet 深度，输入都固定为 520 px。库里的推理图是 torchvision 的膨胀
ResNet FCN，而不是原论文那个基于 VGG、带跳跃连接的 FCN-8s 网络。

LibreYOLO 不训练 FCN：对这个家族调用 `train()` 会抛出 `NotImplementedError`，
上面的[支持层级](/docs/models)把它标为仅推理。已发布的两个检查点（checkpoint）
是 torchvision 自己的、在 COCO 上训练的权重，为 LibreYOLO 的加载器做了转换。

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
