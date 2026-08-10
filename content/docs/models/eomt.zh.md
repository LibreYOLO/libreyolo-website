---
title: EoMT
families:
  - eomt
seo_title: EoMT：预测语义分割、实例分割和全景分割
description: >-
  在 LibreYOLO 里用 EoMT 做语义分割、实例分割和全景分割，跑在一个普通的 DINOv2 vision transformer
  上，不需要解码器。采用 MIT 许可。
lead: >-
  一个建在普通 vision transformer
  上的分割网络，没有专门的像素解码器：加到编码器本身上的额外可学习查询（query）直接预测掩码。LibreYOLO 支持用它做语义分割、实例分割和全景分割。
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - 全景分割 python
  - 实例分割 dinov2
  - 语义分割 预训练模型
last_verified: 1.5.0
snippets:
  predict:
    - label: 语义分割
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) 类别 id
        print(mask.classes)      # 图像中出现的类别 id，已排序
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -seg 后缀会选中实例任务，所以这里不需要
        # 传 task 参数
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: 全景分割
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) 分段 id
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: 语义分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 掩码
        print(metrics["metrics/mAP50-95(B)"])   # 检测框
    - label: 全景分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## 安装

EoMT 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。文件名里的任务后缀
（`-sem`、`-seg`、`-panoptic`）决定任务，`LibreYOLO()` 会从这个文件名把它推断
出来，所以不需要 `task=` 参数。

<code-tabs name="predict" />

语义分割填充 `result.semantic_mask`，它在 `.data` 上是一个由类别 id 组成的
`(H, W)` 数组。实例分割填充 `result.boxes` 和 `result.masks`，形状和其他每个
分割家族返回的一样。全景分割填充 `result.panoptic`：`.data` 上是一张 `(H, W)`
的分段 id 图，另外还有 `.segments_info`，一个由 `{"id", "category_id"}` 字典
组成的列表，每个分段一项。`conf` 过滤查询的选择；`iou` 对语义任务不起作用，
因为它是逐像素取 argmax，没有 NMS 步骤。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

三种编码器尺寸 s/b/l，全部基于 DINOv2。语义检查点（checkpoint）在 ADE20K 上以
512 px 训练；实例和全景检查点在 COCO 上以 640 px 训练，另有第二个以 1280 px
训练的实例检查点。上游只在 l 尺寸上提供 DINOv2 的实例分割权重；s 和 b 只发布了
语义和全景两种。上游存在基于 DINOv3 的 EoMT 变体，但这里没有提供，因为它们依赖
需要申请权限的（gated）非商用 DINOv3 权重。

LibreYOLO 不训练 EoMT：对这个家族调用 `train()` 会抛出 `NotImplementedError`，
上面的[支持层级](/docs/models)把它标为仅推理。

## 验证

`val()` 按任务分发。语义返回 `metrics/mIoU` 和 `metrics/pixel_accuracy`。实例
分割返回和其他分割家族一样的掩码与检测框 mAP 键。全景返回全景质量（Panoptic
Quality），键为 `metrics/PQ`，并拆分成 `metrics/SQ`（分割质量）和
`metrics/RQ`（识别质量），另有 `metrics/PQ_things` 和 `metrics/PQ_stuff`。

<code-tabs name="val" />

## 导出

<export-matrix />

目前只有语义任务能导出：实例分割和全景分割调用 `export()` 会得到
`NotImplementedError`，因为它们的查询掩码输出还没有导出运行时的契约。导出的语义
产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine` 文件的
表现和检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
