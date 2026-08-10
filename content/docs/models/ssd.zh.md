---
title: SSD
families: [ssd]
seo_title: "SSD（SSD300）：在 LibreYOLO 里做目标检测"
description: "在 LibreYOLO 里运行 SSD300：一个单次前向的 VGG16 检测器，采用 BSD-3-Clause 许可，可以预测、验证并导出 ONNX。没有训练路径。"
lead: "SSD（Single Shot MultiBox Detector）在一次前向传播中，从一张稠密的默认框（default box）网格上算出每个检测框和类别分数，没有单独的候选区域生成阶段。LibreYOLO 以仅推理的检测器形式提供基于 VGG16 的 SSD300 检查点。"
keywords: [SSD, SSD300, "Single Shot MultiBox Detector", "ssd 目标检测", "ssd300 推理", VGG16, "基于锚框的检测器"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSSD300.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # 这里故意不写 imgsz：SSD300 会在它检查点的原生画布上追踪，
        # 传任何其他值都会在导出开始前抛出异常
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

SSD 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改
一行的事。SSD 用逐类别分数解码它的默认框网格，然后跑非极大值抑制（non-maximum
suppression），所以 `conf`、`iou` 和 `max_det` 在这里都真的起作用，这一点和本库
里基于 query 的检测器不同。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

SSD 只提供一个检查点：基于 VGG16 的 SSD300 网络，跑在它固定的原生画布上。这个
家族里没有尺寸或规模可选；预测、验证和导出用的都是那一张计算图。

权重文件是 `LibreSSD300.pt`，即家族前缀加上它唯一的尺寸键 `"300"`。它背后的类是
`LibreSSD`，所以直接构造的写法是 `LibreSSD(size="300")`，而不是一个以文件名命名
的类。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

SSD 只能导出到 ONNX；这个家族的其他格式目前全部被屏蔽。导出始终使用检查点的原生
画布，而且计算图暴露的是 SSD 未经处理的打包 head，而不是融合了非极大值抑制的输
出，所以导出时不接受 `nms=True`。LibreYOLO 自己的后端会在把计算图加载回来之后，
再跑解码和抑制这一步。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

LibreYOLO 的 SSD300 代码不是从论文作者自己的 Caffe 版本移植来的；它派生自
torchvision 采用 BSD-3-Clause 许可的 SSD300 实现，上面链接的上游来源就是那个仓
库。骨干的 VGG16 权重还能再往前追溯到牛津大学的全卷积精简版 VGGNet，由 Karen
Simonyan 和 Andrew Zisserman 以 CC-BY-4.0 许可发布。

</provenance-box>

## 引用

<citation-block />
