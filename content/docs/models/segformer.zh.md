---
title: SegFormer
families:
  - segformer
seo_title: SegFormer：在 LibreYOLO 里做语义分割
description: 在 LibreYOLO 里用 SegFormer 做 ADE20K 语义分割，覆盖 b0-b5 六种尺寸。安装、预测、训练并导出；预训练权重是非商用的。
lead: >-
  SegFormer 是一个语义分割 transformer，它把分层的 Mix Transformer（MiT）编码器和一个轻量的 all-MLP 解码
  head 配在一起，避开了早期分割 transformer 需要的笨重解码器和固定位置编码。LibreYOLO 只把它用于一个任务，语义分割，覆盖六种尺寸。
keywords:
  - SegFormer
  - 语义分割 python
  - Mix Transformer
  - MiT
  - transformer 语义分割
  - ADE20K
  - segformer 预训练权重
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python（微调）
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: 从零训练
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # 不传 model_path：随机初始化，什么都不下载。这是拿到不带预训练
        # 检查点非商用条款的权重的唯一途径
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## 安装

SegFormer 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`result.semantic_mask` 携带稠密的类别图：`.data` 是一个按原图尺寸给出的
`(H, W)` 类别 id 张量，`.classes` 列出实际出现的类别 id。`result.boxes` 是
`None`，因为没有逐实例的检测结果。`conf` 和 `iou` 为了 API 一致而被接受，但不会
改变输出：模型给每个像素返回一个类别，而不是需要过滤或去重的逐实例检测。数据源、
流式处理和结果处理见[预测](/docs/predict)。

## 变体

六种尺寸，b0 到 b5，每一步都把 Mix Transformer 编码器加宽、加深，同时保持同样的
all-MLP 解码 head 设计。

<checkpoint-table />

## 训练

`train()` 默认对已发布的检查点（checkpoint）做微调。改为不给
`LibreSegformer(...)` 传 `model_path`，它就会用随机初始化的编码器和 head 来构建，
从零开始训练，这是拿到不带任何预训练检查点非商用限制的权重的唯一途径（见
[许可证](#licensing)）。

<code-tabs name="train" />

不做改动的话，训练器遵循 SegFormer 论文的 ADE20K 配方：AdamW，骨干用一个基础
学习率，解码 head 用它的 10 倍学习率训练，除 LayerNorm 和 Mix-FFN 位置卷积之外
处处施加权重衰减，再配一个带预热的线性衰减调度。更大的几个尺寸，b3 到 b5，收敛性
尚未端到端验证。

数据集、数据增强、多卡训练和日志器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 开头的键组成的字典：mIoU 和像素精度，在任何采用你
训练所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。[导出](/docs/export)列出了每种格式
接受的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

LibreSegformer 的编码器和解码 head 是对 Hugging Face Transformers 中采用
Apache-2.0 许可的 SegFormer 实现的 PyTorch 移植，而不是对 NVlabs/SegFormer 的
移植：英伟达的原始仓库从未被阅读或复制，这里列出它只是为了向论文作者致谢。只有
上面那些预训练检查点带有英伟达的非商用限制；架构和 LibreYOLO 自己的代码自始至终
都是 MIT。

</provenance-box>

## 引用

<citation-block />
