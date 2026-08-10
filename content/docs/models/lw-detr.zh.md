---
title: LW-DETR
families: [lwdetr]
seo_title: "LW-DETR：预测与导出，采用 Apache-2.0 许可"
description: "在 LibreYOLO 里运行 LW-DETR 做实时目标检测。安装、预测、验证并导出五种基于 ViT 的尺寸，全部采用 Apache-2.0 许可。"
lead: "一个纯 ViT 的检测 transformer，百度把它定位成 YOLO 检测器的实时替代方案。LibreYOLO 提供五种尺寸用于检测，仅支持推理。"
keywords: [LW-DETR, "lw-detr 推理", "检测 transformer", 实时目标检测, plain ViT, DETR, "百度 目标检测"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreLWDETRt.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() 返回的是一个普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640
        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

LW-DETR 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行
的改动。`conf` 和 `max_det` 过滤 query 的选择；`iou` 为了 API 一致性会被接受，
但没有任何作用，因为解码器是一个集合预测器，没有 NMS 步骤。数据源、流式处理和
结果处理见[预测](/docs/predict)。

在 LibreYOLO 里 LW-DETR 只支持推理。上游用跨多个 query 组的 Group-DETR 一对多
监督和一个 IoU 感知的分类损失函数来训练；这套配方这里没有接上，所以 `train()`
会抛出 `NotImplementedError`。

## 变体

五种尺寸，共用同一个纯 ViT 编码器、多尺度投影器和 deformable DETR 解码器，输入
分辨率也都相同。最小的两个共用一个编码器宽度，按 block 深度区分；接下来的两个
共用一个更宽的编码器，按有多少个投影器层级喂给解码器区分；最大的那个则换上最宽
的编码器。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何符合你训练所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点（checkpoint）一样，返回同样的 `Results`。[导出](/docs/export)
列出了每种格式接受的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
