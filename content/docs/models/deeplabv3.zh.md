---
title: DeepLabv3
families: [deeplabv3]
seo_title: "DeepLabv3：预测并导出 ASPP 语义分割模型"
description: "在 LibreYOLO 里用 DeepLabv3 做语义分割。安装、预测、验证并导出 torchvision 的 ResNet 和 MobileNetV3 检查点。"
lead: "一个语义分割网络，在给每个像素分类之前，先以多个膨胀率并行池化特征（空洞空间金字塔池化，atrous spatial pyramid pooling）。LibreYOLO 只把它用于语义分割。"
keywords: [DeepLabv3, ASPP, "语义分割 python", "空洞卷积 语义分割", "deeplabv3 预训练权重"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) 类别 id
        print(mask.classes)      # 图像中出现的类别 id，已排序
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeepLabv3r50-sem.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## 安装

DeepLabv3 不需要任何可选 extra。它导入的一切都在基础安装里。

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

三种骨干：膨胀 ResNet-50、膨胀 ResNet-101 和膨胀 MobileNetV3-Large。这是
DeepLabv3，不是 DeepLabv3+，所以没有解码器阶段，也没有 CRF 精修，与
torchvision 的实现一致，而不是论文自带的参考代码。

LibreYOLO 不训练 DeepLabv3：对这个家族调用 `train()` 会抛出
`NotImplementedError`，上面的[支持层级](/docs/models)把它标为仅推理。已发布的
三个检查点（checkpoint）是 torchvision 自己的、带 VOC 标签的 COCO 权重，为
LibreYOLO 的加载器做了转换。

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
