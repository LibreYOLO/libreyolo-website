---
title: SigLIP2
families: [siglip2]
seo_title: "SigLIP2：在 LibreYOLO 里做零样本分类和嵌入向量"
description: "用 LibreYOLO 里的 SigLIP2 做零样本图像分类和图像/文本嵌入向量，按类别独立的 sigmoid 多标签打分。无需训练。"
lead: "SigLIP2 是一个双塔（dual-tower）模型，它拿图像去和文本提示词打分，每个类别用一个独立的 sigmoid，而不是在一个固定的标签集合上共用一个 softmax。LibreYOLO 支持用它做零样本分类和图像/文本嵌入向量，没有训练这一步。"
keywords: [SigLIP2, SigLIP 2, 零样本图像分类, "siglip2 图像嵌入 python", "siglip2 文本嵌入", 开放词汇分类, 多语言图像分类, "sigmoid 多标签分类"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: |
        # 不调用 set_classes() 时，CLI 的 predict 用的是模型默认加载的
        # 1,000 个 ImageNet 类别名
        libreyolo predict model=LibreSigLIP2b16-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 多标签 sigmoid 打分
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # 每个类别的概率互相独立：可以同时有多个类别得分很高，
        # 也可以一个都没有；默认的 softmax 则会把它们归一化成
        # 单标签分布，和 LibreCLIP 的行为一致
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: 图像与文本嵌入向量
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # 两者都做了 L2 归一化，所以直接点积就是余弦相似度
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data 是一个带 train/ 划分的 ImageFolder 根目录，它的文件夹名
        # 会成为本次运行的零样本类别提示词
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # 当前 set_classes() 设置的标签和输入分辨率会被固化进计算图，
        # 改动其中任何一个之后都要重新导出；导出时 multi_label 必须是
        # False，也就是默认值
    - label: CLI
      language: bash
      code: |
        # 这里没有调用 set_classes()，所以固化进去的是模型默认加载的
        # 1,000 个 ImageNet 类别
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: 嵌入向量导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" 只追踪图像塔，不需要类别
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
---

## 安装

SigLIP2 需要它自己的 extra，它会装上其多语言分词器所用到的 SentencePiece 包。

```bash
pip install "libreyolo[siglip2]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`set_classes()` 是让它成为开放词汇分类器的那一个原语：它把每个标签渲染进每一个提示词模板，编码之后取平均，并把得到的 `[K, D]` 矩阵缓存为分类器 head，所以不会每张图像重算一次。随时再调用一次就能更换类别。如果一次都不调用，LibreSigLIP2 加载时就已经设好了 1,000 个 ImageNet-1k 类别名。

SigLIP 对每个类别独立打分：`logit = scale * (image . text) + bias`。默认情况下，这组 logit 仍然会过一次 softmax，给出一个单标签分布，和 LibreCLIP 的 `top1`/`top5` 行为一致。给 `set_classes()` 传 `multi_label=True`（或者在构造时传），就会换成互相独立的 sigmoid 概率，于是同一张图像上可以有多个类别得分很高，也可以一个都没有。分词器是一个多语言的 SentencePiece 模型（Gemma 词表），所以用英语以外的语言写类别名，效果是一样的。

用 `task="embed"` 时，预测对每个输入返回一个经过 L2 归一化的图像向量，而不是类别概率；`embed_text()` 返回同一个向量空间里归一化的文本行，所以两者之间直接点积就是余弦相似度。`iou` 对这两个任务都没有影响，也没有 NMS 这一步。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 验证

`val()` 读取 ImageFolder `train/` 划分下的类别文件夹名，用它们调用 `set_classes()`，然后在 softmax 打分下测量零样本的 top-1 和 top-5 精度。精度取决于这些类别名当作提示词读起来怎么样，而不是取决于任何权重更新，因为这里根本没有东西可训练。验证只覆盖 `task="classify"`，`task="embed"` 没有数据集验证器。

<code-tabs name="val" />

## 导出

<export-matrix />

导出会把模型当前的状态固化成一张静态计算图。对 `task="classify"` 来说，`set_classes()` 最后设置的那批标签，以及导出时的分辨率，会连同学到的 scale 和 bias 一起固化进最后一层线性层，所以导出的计算图就是一个普通的 `[B, K]` 图像分类器，没有文本塔，也没有分词器；改动类别或尺寸之后要重新导出。`multi_label=True` 模式下的导出没有实现，先把它设回 `False`。`task="embed"` 的导出只追踪图像塔。两者都需要 ONNX opset 14 或更高，导出器默认就设成了这个值。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。两个都是从 Google 采用 Apache-2.0 许可的 `siglip2-base-patch16-256` 和 `siglip2-so400m-patch14-384` 检查点（checkpoint）转换而来，而不是来自任何一次 COCO 训练。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
