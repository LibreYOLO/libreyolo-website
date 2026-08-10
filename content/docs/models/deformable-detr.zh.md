---
title: Deformable DETR
families: [deformable_detr]
seo_title: "Deformable DETR：预测与导出，采用 Apache-2.0 许可"
description: "在 LibreYOLO 里运行 Deformable DETR 做目标检测。安装、预测、验证并导出五种稀疏注意力尺寸，全部采用 Apache-2.0 许可。"
lead: "Deformable DETR 把 DETR 的稠密交叉注意力换成了围绕每个参考点的稀疏、多尺度采样，正是这一点让 transformer 检测器的训练变得可行。LibreYOLO 提供五种尺寸用于检测，仅支持推理。"
keywords: [Deformable DETR, "deformable detr 推理", "检测 transformer", 稀疏注意力, 多尺度可变形注意力, "商汤 目标检测"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeformableDETRr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() 返回的是一个普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800
        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt imgsz=800 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

Deformable DETR 不需要任何可选 extra。它导入的一切都在基础安装里，用的是纯
PyTorch 的多尺度可变形注意力核心。

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

在 LibreYOLO 里 Deformable DETR 只支持推理。上游用匈牙利匹配和 focal 分类损失
函数来训练；这套配方这里没有实现，所以 `train()` 会抛出 `NotImplementedError`。

## 变体

五个检查点（checkpoint）覆盖了已发布的各种配置，输入分辨率都相同。`r50ss` 把
注意力限制在单一特征尺度上；`r50ssdc5` 在此基础上再加一个空洞 C5 骨干阶段。
`r50` 是默认的多尺度配置，在四个特征图层级上采样。`r50refine` 在各个解码器层
之间加入迭代式的检测框精修，`r50twostage` 则从编码器输出而不是学习得到的 query
生成初始的区域候选。

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

<provenance-box></provenance-box>

## 引用

<citation-block />
