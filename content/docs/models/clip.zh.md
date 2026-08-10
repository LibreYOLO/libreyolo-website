---
title: CLIP
families: [clip]
seo_title: "CLIP：在 LibreYOLO 里做零样本分类和嵌入向量"
description: "用 LibreYOLO 里的 CLIP 做零样本图像分类和图像/文本嵌入向量。无需训练：set_classes() 在运行时定义标签集合。"
lead: "CLIP 是一个双塔（dual-tower）模型，它拿图像去和文本提示词打分，而不是去对一个固定的标签集合。LibreYOLO 支持用它做零样本分类和图像/文本嵌入向量，没有训练这一步。"
keywords: [CLIP, OpenCLIP, 零样本图像分类, "clip 图像嵌入 python", "clip 文本嵌入", "图文相似度检索", 开放词汇分类, LAION-2B]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: |
        # 不调用 set_classes() 时，CLI 的 predict 用的是模型默认加载的
        # 1,000 个 ImageNet 类别名
        libreyolo predict model=LibreCLIPb32-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 图像与文本嵌入向量
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # 两者都做了 L2 归一化，所以直接点积就是余弦相似度
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data 是一个带 train/ 划分的 ImageFolder 根目录，它的文件夹名
        # 会成为本次运行的零样本类别提示词
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # 当前 set_classes() 设置的标签和输入分辨率会被固化进计算图，
        # 改动其中任何一个之后都要重新导出
    - label: CLI
      language: bash
      code: |
        # 这里没有调用 set_classes()，所以固化进去的是模型默认加载的
        # 1,000 个 ImageNet 类别
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: 嵌入向量导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" 只追踪图像塔，不需要类别
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
---

## 安装

CLIP 需要它自己的 extra，它会装上其自带的 BPE 分词器复现完全一致的 token id 所用到的包。

```bash
pip install "libreyolo[clip]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`set_classes()` 是让它成为开放词汇分类器的那一个原语：它把每个标签渲染进每一个提示词模板，编码之后取平均，并把得到的 `[K, D]` 矩阵缓存为分类器 head，所以不会每张图像重算一次。随时再调用一次就能更换类别。如果一次都不调用，LibreCLIP 加载时就已经设好了 1,000 个 ImageNet-1k 类别名。

用 `task="embed"` 时，预测对每个输入返回一个经过 L2 归一化的图像向量，而不是类别概率；`embed_text()` 返回同一个向量空间里归一化的文本行，所以两者之间直接点积就是余弦相似度。`iou` 对这两个任务都没有影响，也没有 NMS 这一步。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 验证

`val()` 读取 ImageFolder `train/` 划分下的类别文件夹名，用它们调用 `set_classes()`，然后测量零样本的 top-1 和 top-5 精度。精度取决于这些类别名当作提示词读起来怎么样，而不是取决于任何权重更新，因为这里根本没有东西可训练。验证只覆盖 `task="classify"`，`task="embed"` 没有数据集验证器。

<code-tabs name="val" />

## 导出

<export-matrix />

导出会把模型当前的状态固化成一张静态计算图。对 `task="classify"` 来说，`set_classes()` 最后设置的那批标签，以及导出时的分辨率，都会被固化进最后一层线性层，所以导出的 ONNX 或 TensorRT 计算图就是一个普通的 `[B, K]` 图像分类器，没有文本塔，也没有分词器；改动类别或尺寸之后要重新导出。`task="embed"` 的导出只追踪图像塔。两者都需要 ONNX opset 14 或更高，导出器默认就设成了这个值。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。两个都是从 OpenCLIP 用 LAION-2B 训练的检查点（checkpoint）`ViT-B-32` 和 `ViT-B-16` 转换而来，而不是来自任何一次 COCO 训练。

<checkpoint-table />

LAION-2B 的训练数据有一段被记录在案的 CSAM 内容历史（Stanford Internet Observatory，2023 年 12 月）。LAION 此后发布了 Re-LAION，一个清理过的重新发布版本；如果你要进一步转存这些权重，在有可用版本时优先选择基于 Re-LAION 的检查点。

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
