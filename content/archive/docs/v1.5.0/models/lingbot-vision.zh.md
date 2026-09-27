---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: LingBot-Vision：LibreYOLO 里的语义分割
description: >-
  在 LibreYOLO 里用 LingBot-Vision 做语义分割，跑在采用 Apache-2.0 许可的 ViT
  骨干上。安装、预测、训练、验证和导出，尺寸有 s/b/l。
lead: >-
  LingBot-Vision 是 Robbyant 发布的一系列自监督 vision transformer
  骨干，用以边界为中心的掩码建模训练，面向稠密空间感知。LibreYOLO 给这个骨干配上一个稠密 head，只支持一个任务：语义分割。
keywords:
  - LingBot-Vision
  - 语义分割 python
  - vision transformer 语义分割
  - 自监督预训练 骨干
  - lingbot-vision 权重
  - Robbyant
  - 稠密预测
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python（线性探针）
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 骨干默认冻结，与上游的评测协议一致：
        # 只训练 1x1 的稠密 head
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: 完整微调
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## 安装

LingBot-Vision 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`result.semantic_mask` 承载稠密的类别图：`.data` 是一个按原图尺寸给出的
`(H, W)` 类别 id 张量，`.classes` 列出实际出现的类别 id。`result.boxes` 是
`None`，因为这里没有逐实例的检测结果。`conf` 和 `iou` 为了 API 一致而被接受，
但不会改变输出，因为模型返回的是每个像素一个类别，而不是可供过滤的检测结果。
数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

已发布三种尺寸，s、b 和 l，从一个 11 亿参数的 ViT-g/16 教师模型蒸馏而来。教师
模型本身，也就是尺寸 `g`，在 LibreYOLO 里可以加载和微调，但 LibreYOLO 自己并不
托管 `g` 检查点（checkpoint）。

<checkpoint-table />

## 训练

`train()` 微调一个已发布的检查点。默认配方是上游报告里的线性探针：ViT 骨干被
冻结，只训练 1x1 的稠密 head，与上面那些由 LibreYOLO 托管的权重的产出方式一致。
传 `freeze_backbone=False` 可以改成微调整个网络，并且要相应调低 `lr0`。

<code-tabs name="train" />

数据集、数据增强、多卡训练和日志器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 开头的键组成的字典：mIoU 和像素精度，在任何采用你
训练所用格式的数据集上测量。

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

上游发布的说明写明，它的 ViT 建立在 Meta AI 发表的 DINOv2/DINOv3 架构之上。
Robbyant 以 Apache-2.0 许可分发他们的实现，而这次 LibreYOLO 的移植只依据
Robbyant 的仓库完成，从未取自 Meta 的 DINOv2 或 DINOv3 代码。

</provenance-box>

## 引用

<citation-block />
