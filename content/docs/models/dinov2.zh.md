---
title: DINOv2
families:
  - dinov2
seo_title: LibreYOLO 里的 DINOv2：语义分割、分类与嵌入
description: >-
  在 LibreYOLO 里用 DINOv2 做语义分割、分类和整图嵌入，全部跑在 DINOv2-with-Registers 骨干上。从头到尾采用
  Apache-2.0 许可。
lead: >-
  DINOv2 是 Meta AI 训练的自监督 vision transformer，不用标注就能产出通用的图像特征。LibreYOLO 把它的
  DINOv2-with-Registers 骨干封装成三个任务：语义分割、分类和整图嵌入。
keywords:
  - DINOv2
  - DINOv2 with registers
  - 自监督学习
  - dinov2 特征提取
  - vision transformer 语义分割
  - 图像 embedding 提取
  - 以图搜图 特征向量
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: 语义分割
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # 这个家族没有 LibreYOLO 托管的检查点：这里会从 Meta 的 Hugging Face
        # 组织下载采用 Apache-2.0 许可的 DINOv2-with-Registers-small 骨干。
        # 稠密 head 在你训练它之前一直是随机初始化的（见下面的训练一节）
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: 分类
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= 是你数据集的类别数；线性 head 在你训练它之前一直是
        # 随机初始化的
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: 嵌入
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # 绕过所有任务 head：光靠骨干就够了，所以不用微调就能用
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D)，已做 L2 归一化
    - label: 批量嵌入
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # 便捷封装：内部跑一次 predict()，把每一行堆叠成一个 (N, D) 张量
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: 语义分割
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: 分类
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: 多卡训练
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: 语义分割
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: 分类
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: 语义分割
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: 分类
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: 嵌入
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，返回的
        # 也是同一个 Results 对象。导出时文件名由任务决定，这里是
        # LibreDINOv2s-sem.onnx
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## 安装

LibreDINOv2 只在装了 `transformers` 之后才会注册，也就是 RF-DETR 给它的 DINOv2
骨干所需要的那个可选依赖，所以它用的是同一个 extra。

```bash
pip install "libreyolo[rfdetr]"
```

## 预测

LibreYOLO 没有发布 LibreDINOv2 检查点（checkpoint）。不要去加载文件，直接构造这个
封装：`model_path=None`（默认值）会在首次使用时从 Hugging Face 下载 Meta 采用
Apache-2.0 许可的 `facebook/dinov2-with-registers-small` 骨干。`task=` 选择在它之上
跑什么。

<code-tabs name="predict" />

`task="semantic"` 和 `task="classify"` 会在骨干之上加一个稠密 head 或线性 head；
这个 head 是随机初始化的，只有在你训练它之后才有用（见[训练](#train)）。
`task="embed"` 跳过所有 head，把骨干最后归一化的 CLS token 作为整图的一行返回到
`result.embeddings` 里，所以它完全不需要训练。`result.boxes` 永远是 `None`：这三个
任务都不产生逐实例的检测结果。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

`size` 选的是叠在骨干之上的、RF-DETR 风格的投影层宽度，而不是骨干本身：每种尺寸共用
同一个 DINOv2-S（small）编码器。语义分割跑在 DINOv2 原生的方形 patch 网格上；分类和
嵌入跑在训练线性探针时用的那个更小的分类分辨率上。

## 训练

`task="semantic"` 和 `task="classify"` 都能训练；`task="embed"` 没有依赖类别的 head
要拟合，你在它上面调用 `train()` 会抛出 `NotImplementedError`。

<code-tabs name="train" />

这里的主要关键字参数是 `batch_size` 和 `lr`，不是大多数其他家族用的 `batch` 和
`lr0`；`batch` 和 `lr0` 仍然接受，并会映射到前者上，但两个一起传会报冲突错误。
`output_dir=`（默认 `"runs/train"`）取代 `project=`/`name=`，成为安放一次运行结果的
主要方式，不过直接传 `project=`/`name=` 也仍然有效。数据集、数据增强、多卡训练和
日志器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 开头的键组成的字典：`task="semantic"` 是 mIoU 和像素
精度，`task="classify"` 是 top-1 和 top-5 精度。`task="embed"` 没有可供打分的真值
（ground truth），你在它上面调用 `val()` 会抛出 `NotImplementedError`。

<code-tabs name="val" />

## 导出

<export-matrix />

每个任务支持的格式子集不同，如上所示。导出的产物按文件后缀通过 `LibreYOLO()` 加载
回来，所以一个 `.onnx` 或 `.engine` 文件的表现和检查点一样，返回同样的 `Results`。
[导出](/docs/export)列出了每种格式接受的参数。

<code-tabs name="export" />

## 许可证

<provenance-box>

上面「权重」那一行写的是适用的许可证，Apache-2.0，但这个家族其实没有任何东西以
LibreYOLO 的 Hugging Face 组织名义重新发布：LibreYOLO 自己并不托管 LibreDINOv2 检查
点。`LibreDINOv2(model_path=None)` 下载的是 Meta 自己的
`facebook/dinov2-with-registers-small` 仓库，原封不动。

</provenance-box>

## 引用

<citation-block />
