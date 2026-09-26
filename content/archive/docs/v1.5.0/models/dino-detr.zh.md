---
title: DINO-DETR
families:
  - dinodetr
seo_title: DINO-DETR：预测与导出，采用 Apache-2.0 许可
description: 在 LibreYOLO 里运行 DINO-DETR 做目标检测。安装、预测、验证并导出三种去噪锚框尺寸，全部采用 Apache-2.0 许可。
lead: >-
  DINO-DETR 由 IDEA Research 以 DINO 之名发布，在 Deformable DETR 的稀疏注意力之上，把对比式去噪训练和混合
  query 选择结合在一起。LibreYOLO 提供三种尺寸用于检测，仅支持推理。
keywords:
  - DINO-DETR
  - DINO
  - dino detr 目标检测
  - 检测 transformer
  - 去噪锚框
  - 混合 query 选择
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() 返回的是一个普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## 安装

DINO-DETR 不需要任何可选 extra。它导入的一切都在基础安装里，用的是和 LibreYOLO
的 Deformable DETR 家族相同的纯 PyTorch 多尺度可变形注意力核心。

```bash
pip install libreyolo
```

安装 `libreyolo[hub-kernels]` 是可选的。一旦 `kernels` 包存在，LibreYOLO 会在
运行时从 Hugging Face Hub 拉取一个编译好的多尺度可变形注意力 kernel，用它替代
纯 PyTorch 核心；`LIBREYOLO_HUB_KERNELS=0` 可以把它再关掉。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行
的改动。`conf` 和 `max_det` 过滤 query 的选择；`iou` 为了 API 一致性会被接受，
但没有任何作用，因为解码器是一个集合预测器，没有 NMS 步骤。数据源、流式处理和
结果处理见[预测](/docs/predict)。

在 LibreYOLO 里 DINO-DETR 只支持推理。上游用对比式去噪和匈牙利匹配来训练；这套
配方这里没有实现，所以 `train()` 会抛出 `NotImplementedError`。

## 变体

三个检查点（checkpoint），输入分辨率都相同。`r50` 和 `r50s5` 共用一个 ResNet-50
骨干，区别在于有多少个特征图尺度送进解码器，四个对五个。`swinl` 把骨干换成
Swin-L，同样采样五个尺度。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何符合你训练所用格式的数据集上测量。

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

<provenance-box>

三个官方检查点来自作者的 Google Drive 发布文件夹，而不是 Hugging Face 模型卡。
上游仓库在仓库层面声明了 Apache-2.0，但没有给检查点本身附上许可证文件或许可证
元数据，所以再分发的依据是那条仓库层面的声明，而不是针对检查点的单独许可。每个
LibreYOLO 镜像都随附上游 Apache-2.0 许可证原文，以及一份解释这一点的说明。

</provenance-box>

## 引用

<citation-block />
