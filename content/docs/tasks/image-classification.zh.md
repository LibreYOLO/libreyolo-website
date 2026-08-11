---
title: 图像分类
seo_title: LibreYOLO 中的图像分类
description: >-
  在 LibreYOLO 中为整张图像打标签：服务这一任务的模型家族、ImageFolder 数据集布局，以及 predict、train、validate
  和 export 调用。
lead: 图像分类为整张图像给出一个标签分布，不定位图像内部的任何东西。任务键是 classify。
keywords:
  - 图像分类 python
  - 训练图像分类模型
  - ImageFolder 数据集
  - top-1 准确率
  - 零样本图像分类
  - MIT 许可 图像分类库
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名中的 -cls 后缀会选定任务，所以不需要额外的
        # task 参数
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 完整的分布
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data 是完整的 (C,) 向量，top5/top5conf 是排好序的视图
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 零样本，无需训练
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP 用文本提示给图像打分，所以标签集在调用时指定，

        # 而不是固化在检查点里

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 是已知的数据集名称，首次使用时自动下载
        # 用自己的数据时，传入一个带 train/ 划分的目录
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() 返回一个普通的 dict，不是对象
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出的产物加载起来和
        # 检查点一样，返回同样的 Results 对象
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## 定义

图像分类为整张图像的每个类别给出一个分数，完全不给坐标。它回答画面里有什么，
而绝不回答在哪里，这正是它与[目标检测](/docs/tasks/object-detection)的区别。

`classify` 是规范的任务键，检查点（checkpoint）文件名中的 `-cls` 后缀会选中它。
在分类家族上这个后缀是必需的，而不是可选的，所以 `LibreResNet50.pt` 不会被当作
分类器读取，只有 `LibreResNet50-cls.pt` 才会。

`predict()` 会填充 `result.probs`，并把 `boxes` 留空。`.data` 是完整的分数向量，
`.top1` 是最高分的索引，`.top1conf` 是它的值，`.top5` 是按降序排列的五个最高索引，
`.top5conf` 是它们的分数。索引指向 `result.names`。对 `Results` 对象切片永远不会
截断 `probs`，因为这个向量属于整张图像，而不属于其中某一行。

## 模型

有五个家族既能训练也能预测：[ResNet](/docs/models/resnet)、
[ConvNeXt](/docs/models/convnext)、[MobileNetV4](/docs/models/mobilenetv4)、
[EfficientNetV2](/docs/models/efficientnetv2) 和
[DINOv2](/docs/models/dinov2)。前四个在基础包上就能运行，并提供已发布的权重。
DINOv2 需要 `pip install "libreyolo[rfdetr]"`，而且没有 LibreYOLO 托管的检查点：
它加载上游骨干，配一个随机初始化的线性 head，所以它是微调的起点，而不是一个
开箱即用的预测器。

另有五个能预测、验证和导出，但它们的 `train()` 会抛出
`NotImplementedError`：[ViT](/docs/models/vit)、[Swin](/docs/models/swin)、
[VGG](/docs/models/vgg)、[AlexNet](/docs/models/alexnet) 和
[DeiT](/docs/models/deit)。

[CLIP](/docs/models/clip) 和 [SigLIP2](/docs/models/siglip2) 在没有固定标签集的
情况下分类。它们拿图像和文本提示打分，所以 `set_classes()` 在调用时定义类别，
换一套新的标签集根本不存在训练这一步。两者也服务于 `embed` 任务。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`conf`、`iou` 和 `max_det` 在这里没有作用：没有候选需要用阈值筛选或抑制，只有
一个分布。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 数据集格式

分类使用一棵目录树，而不是标注文件，也不是 YAML。`data` 就是数据集根目录。

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

训练需要 `train/`，它按排序后的文件夹名定义类别到索引的映射，所以字母序上的第一个
文件夹成为类别 0。验证需要 `val/`。可以有 `test/` 划分，默认的训练和验证命令不会
用到它。除 `train` 以外的任何划分都必须包含与预期类别集相同的类别文件夹名，正是
这一点让不匹配直接报错，而不是被算成一次错误的预测。接受的图像扩展名是 `.jpg`、
`.jpeg`、`.png`、`.bmp`、`.webp`、`.tif` 和 `.tiff`。

`data` 接受三种东西：一个包含 `train/` 划分的目录路径、一个 `.zip` URL，或者已知
数据集名称之一，即 `imagenette160` 和 `smoke10`，它们会在首次使用时下载并缓存。

规范的加载器是 `libreyolo.data.classify_dataset`。

## 训练

<code-tabs name="train" />

不需要声明 `nc`：类别数来自 `train/` 下的文件夹名，最后的线性层会被重建以匹配它，
而骨干原样迁移过来。数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的普通字典，在数据集根目录的 `val/` 划分上
计算。

<code-tabs name="val" />

`metrics/accuracy_top1` 是最高分类别正好是真实类别的图像占比，它是首要数字，训练
用它来挑选最佳轮次。`metrics/accuracy_top5` 是真实类别出现在得分最高的五个类别中
任意位置的图像占比，数据集的类别越少，它说明的问题就越少。字典里还带有
`fitness`，它是 top-1 值的一份副本。

## 导出

<code-tabs name="export" />

导出的产物会按文件后缀通过 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点一样，返回同样的 `Results`。格式覆盖范围因家族而异；每个模型
页面上的矩阵是从已验证的集合生成的，而不是手工敲进去的。各种格式、它们的额外
依赖和限制见[导出与部署](/docs/export)。
