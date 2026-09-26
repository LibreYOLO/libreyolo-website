---
title: 语义分割
seo_title: LibreYOLO 中的语义分割
description: 在 LibreYOLO 中为每个像素打上类别标签：服务这个任务的模型家族、稠密掩码格式，以及预测、训练、验证和导出调用。
lead: 语义分割为图像的每个像素分配一个类别，并且不区分同一类别的不同实例。任务键是 semantic。
keywords:
  - 语义分割 python
  - 像素级分类
  - 稠密预测
  - 训练语义分割模型
  - mIoU
  - MIT 许可 语义分割库
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -sem 后缀会选中任务，所以不需要传 task 参数
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # 原图画布上的 (H, W) 类别 id
        print(mask.classes)      # 出现过的类别 id，已排序，忽略 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 一次处理一个类别
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # 布尔型 (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 换个家族，同样的调用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
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
    - label: 在 ADE20K 上
      language: bash
      code: |
        # ade20k.yaml 内嵌了下载脚本，用来获取约 1 GB 的压缩包，
        # 所以除非数据已经在本地，否则需要显式授权
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() 返回一个普通的 dict，不是对象
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
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出的产物加载起来和
        # 检查点一样，返回同样的 Results 对象
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## 定义

语义分割标注的是像素，不是物体。每个像素都得到一个类别 id，图像里相互接触的两辆车
会变成 car 类的一整块区域，中间没有边界。数出实例个数是[实例分割](/docs/tasks/instance-segmentation)；
既标注每个像素、又同时把实例分开，则是[全景分割](/docs/tasks/panoptic-segmentation)。

`semantic` 是规范的任务键，检查点（checkpoint）文件名中的 `-sem` 后缀会选中它，
所以加载已发布的权重时不需要 `task=`。

`predict()` 会填充 `result.semantic_mask`。`.data` 是原图画布上的一张 `(H, W)`
整数类别图，`.classes` 按排序列出出现过的 id，`.class_mask(id)` 返回某一个类别对应
的布尔 `(H, W)` 选择。`255` 这个值是忽略标签：它永远不是一个类别，在损失函数和指标
里都被排除，`.classes` 也不会把它列出来。

## 模型

有三个家族既能训练也能预测：[SegFormer](/docs/models/segformer)、
[LingBot-Vision](/docs/models/lingbot-vision) 和
[DINOv2](/docs/models/dinov2)。SegFormer 和 LingBot-Vision 在基础包上就能运行，
并提供已发布的权重。DINOv2 需要 `pip install "libreyolo[rfdetr]"`，而且没有
LibreYOLO 托管的检查点：它加载上游骨干，稠密 head 从随机初始化开始，所以它是训练的
起点，而不是一个开箱即用的预测器。

另有四个能预测、验证和导出，但它们的 `train()` 会抛出
`NotImplementedError`：[FCN](/docs/models/fcn)、
[DeepLabv3](/docs/models/deeplabv3)、[PIDNet](/docs/models/pidnet) 和
[EoMT](/docs/models/eomt)。

类别集合因检查点而异，不因家族而异。已发布的权重来自标签空间几乎没有共同点的数据集，
其中就有 ADE20K 的 150 个类别对上 Cityscapes 的 19 个，所以告诉你一个检查点能标注
什么的是它的 `names`，而两个检查点只有在同一个标签空间上训练过时才可比。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

这张图是逐像素取 argmax 得到的，所以没有 NMS 这一步，`iou` 永远不起作用。`conf` 和
`max_det` 是为了 API 一致才接受的，在 SegFormer、PIDNet 和其他稠密预测器上什么都不
做；EoMT 是例外，那里 `conf` 会过滤 query 的选择。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 数据集格式

每张图像配的是一张稠密的单通道掩码，而不是 `.txt` 标注文件，它通过把图像路径里的
`images` 换成掩码目录来定位。

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

掩码是无损的单通道图像，通常是 PNG，调色板模式的 PNG 会按调色板索引读取。每个像素值
都是 `0..nc-1` 里的一个类别 id，`255` 表示忽略，掩码的分辨率必须与配对图像的分辨率
一致。

YAML 在共用约定之外还接受两个键：

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` 是用来替换 `images` 的目录名，默认为 `masks`。`label_mapping` 是可选的
`{source_id: train_id}` 重映射，在加载时作用于掩码的像素值，编号从 1 到 150 的数据集
就是这样变成 0 到 149 的；任何没有被映射的源值都会变成忽略，而每个 train id 都必须
落在 `0..nc-1` 内。

不写 `masks_dir` 会让加载器切换到一条兜底路径：掩码在加载时由多边形标注栅格化而来，
这些标注按通常的 `images` 到 `labels` 约定解析，并且会在物体类别之后追加一个
`background` 类，所以 `nc` 会加一。

规范的加载器是 `libreyolo.data.SemanticDataset`。

## 训练

<code-tabs name="train" />

这里的 `imgsz` 受到检测器上没有的约束。每个家族都声明了一个除数，输入必须是它的整数
倍，这个除数由该家族的 patch 网格或输出步长决定，`imgsz` 除不尽时，训练和验证都会在
运行开始前抛出 `ValueError`。SegFormer 的除数是 32，LingBot-Vision 和 EoMT 是 16，
DINOv2 是 14，FCN 和 PIDNet 是 8。数据集、数据增强、多卡训练和日志记录器见
[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的普通字典，在数据集 YAML 中由 `val` 指定的划分
上计算。

<code-tabs name="val" />

`metrics/mIoU` 是平均交并比：对每个类别，取预测像素与真实像素的重叠除以它们的并集，
再在类别之间求平均。它是首要数字，训练期间也用它来挑选最佳轮次。
`metrics/pixel_accuracy` 是被给出正确类别的像素占比，一个面积很大的背景类可以把它
抬高，所以拿来对比的数字应该是 mIoU。标为 `255` 的像素两者都不计入。字典里还带有
`fitness`，它是 mIoU 值的一份副本。

## 导出

<code-tabs name="export" />

导出的产物会按文件后缀通过 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点一样，返回同样的 `Results`。格式覆盖范围因家族而异；每个模型页面上
的矩阵是从已验证的集合生成的，而不是手工敲进去的。各种格式、它们的额外依赖和限制见
[导出与部署](/docs/export)。
