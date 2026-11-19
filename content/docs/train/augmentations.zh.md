---
title: 数据增强
seo_title: LibreYOLO 中的训练数据增强，附示例
description: LibreYOLO 每个数据增强参数的作用：由库自己的训练代码在一张真实图像上演示，并列出它的取值范围、默认值以及读取它的家族。
lead: >-
  训练数据增强是施加在每个训练样本上的随机变换。它们由 TrainConfig
  上的参数配置，而每个模型家族都跑自己的流水线，所以一个参数只有在该流水线读取它时才会改变样本。
keywords:
  - yolo 数据增强
  - mosaic 增强
  - mixup
  - hsv 数据增强
  - 随机仿射变换
  - 随机擦除 random erasing
  - 小目标检测
  - 数据增强 示例
  - copy paste 数据增强
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # CLI 把 mosaic_prob 写作 mosaic，把 mixup_prob 写作 mixup
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX 从 mosaic_scale=(0.1, 2.0) 起步，可能把目标缩小到原尺寸的
        # 十分之一；(0.5, 1.5) 最多缩小到一半
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # 在 CLI 上，一对数值写成两个用逗号分隔的数字
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: 读取某个家族的支持表
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: 只看被忽略的
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: 分类参数包
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 72d22a38d647de0a
---

## 设置这些参数

数据增强参数就是普通的 `train()` 参数。

<code-tabs name="train" />

其中三个有更短的写法，`train()` 和 CLI 都接受：`mosaic` 对应 `mosaic_prob`，`fliplr` 对应 `flip_prob`，在检测上 `mixup` 对应 `mixup_prob`。如果同时传入简短写法和对应字段，且两者取值不同，会报错。`perspective` 在 1.8.0 中没有 CLI 标志，所以要在 Python 中设置。

分类参数会在构建其流水线时校验，CLI 还会检查 `flip_prob`、`flipud`、`cutmix` 和 `erasing`。`TrainConfig` 不对检测参数做范围检查，所以 `degrees=100` 会以最多 100 度的旋转来训练。[数据增强矩阵](/docs/reference/augmentation-matrix)列出了每个参数的取值范围。

## 三种状态，而不是两种

一个参数是否起作用，取决于家族。库里维护着一张描述这一点的声明式表格，每一项都是三种状态之一。

`used` 表示参数会到达流水线并改变样本。`ignored` 表示它永远到不了流水线，所以设置它没有任何效果。`gated_by_mosaic` 表示它只作用于走了 mosaic 分支的样本，所以在 `mosaic_prob=0` 时，即使它已经接入，也永远不会触发。

第三种状态正是让人意外的那一种。在 YOLOX 风格的流水线上，仿射变换跑在 mosaic 画布上，MixUp 混合的也是 mosaic 样本，所以 `mosaic_prob=0` 会一次性悄悄关掉 `degrees`、`translate`、`shear`、`perspective`、`mosaic_scale`、`mixup_prob` 和 `mixup_scale`。训练器专门为 MixUp 这种情况记录一条警告：

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI 也会对被忽略的参数发出警告，并且只列出你真正输入过的那些：

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

从 Python 调用时，`train()` 不会对被忽略的参数发出警告。[数据增强矩阵](/docs/reference/augmentation-matrix)给出了每个家族中每个参数的状态，以及每个家族的起始值。这张表就是随库分发的 spec，位于 `libreyolo/data/augment/spec.py`，库自己的测试会拿它和流水线做核对，也可以直接读取它：

<code-tabs name="support" />

## 每个参数的作用

下面的图由 LibreYOLO 1.8.0 自己的训练变换生成，通过一个单图数据集在示例图像上运行，所以每个 mosaic 拼块和每个 MixUp 配对样本都是同一张照片。检测框标出的是 LibreYOLO9t 在这张图上检测到的那个人。每张子图下方标出所用的设置和最大检测框的尺寸，单位是 640 像素训练分辨率下的像素。不做数据增强时，这个人的尺寸是 291 × 217 px。

有几个几何参数是范围：`degrees=10` 表示角度在 -10 到 +10 度之间任意抽取。它们的子图把抽取值固定为图注中的值，所以展示的是该设置所能产生的最强效果，而不是一个典型样本。检测相关的图来自 YOLOv9 流水线。YOLOv7 原样经由它训练，RT-DETR 和 RT-DETRv2 使用它的 mosaic 包装器，再配上各自的逐样本变换。YOLOX、RTMDet、PicoDet 和 FOMO 在一个单独的包装器里以同样的方式构建 mosaic 和仿射变换，而这个包装器的 MixUp 有所不同。

### mosaic_prob

一个样本成为 mosaic 的概率：四张图像放在一块两倍于输入尺寸的画布上，再由仿射变换把它缩放到输入尺寸。在 YOLOX 风格的流水线上，仿射变换和 MixUp 只作用于 mosaic 样本，所以这个值为 0 时两者都会停止。RT-DETR、RT-DETRv2、PicoDet 和 FOMO 默认将它设为 0。

<aug-figure name="mosaic" />

第二张子图把仿射变换的缩放固定在 0.5×，此时整块画布恰好能完整放下。最后两张是使用 YOLOv9 默认值的普通抽取，图注给出了各自抽到的缩放值。

### mosaic_scale

仿射变换缩放倍数的抽取范围。在 1× 时，目标保持它在推理时的大小；在 0.5× 时，则是推理时的一半。基础默认值是 `(0.1, 2.0)`，YOLOX、YOLOv7、RTMDet、PicoDet、RT-DETR、RT-DETRv2 和 FOMO 沿用这个值，所以它们的目标可能缩小到原尺寸的十分之一。YOLOv9 默认使用 `(0.5, 1.5)`。YOLO-NAS 和 PP-YOLOE 没有 mosaic，它们把这个范围复用为各自逐样本仿射变换的缩放范围。

<aug-figure name="mosaic_scale" />

高于 1× 时，输出只显示画布左上角的一部分，所以目标会在边缘被切断，或者移出画面。

### degrees

旋转范围：每个经过仿射变换的样本会转过一个在 `-degrees` 到 `+degrees` 之间抽取的角度。旋转以画布的左上角为支点，所以角度一大，图像的大部分就会转出画面。YOLOX 和 YOLOv7 沿用基础默认值 10，YOLOv9 默认使用 0。

<aug-figure name="degrees" />

### translate

平移范围，以输入边长的比例表示：在 `imgsz=640` 时，每个经过仿射变换的样本在每个轴上最多移动 `translate` 乘以 640 px。基础默认值是 0.1。YOLO-NAS 和 PP-YOLOE 默认使用 0.25，RTMDet、PicoDet 和 FOMO 默认使用 0。

<aug-figure name="translate" />

### shear

错切范围，单位为度，每个轴分别抽取。YOLOX 和 YOLOv7 沿用基础默认值 2，其他所有读取它的家族默认使用 0。

<aug-figure name="shear" />

### perspective

叠加在旋转、缩放、错切和平移之上的投影变换。画布的每个角点在每个轴上向内移动一个随机量，最多为 `min(perspective * 100, 0.2)` 乘以画布边长的一半，所以 `perspective=0.001` 在任何分辨率下最多把一个角点移动边长的 5%，而从 `0.002` 起的每个取值都会达到同一个最大值。默认值 `0` 让变换保持为仿射，负值会在第一个样本做变换时抛出 `ValueError`。这些子图共用同一次随机抽取，只按取值缩放。YOLOv9 及其 E2E 和 P2 变体、YOLOX、YOLOv7、YOLO-NAS、PP-YOLOE、RTMDet、PicoDet、RT-DETR 和 RT-DETRv2 都会应用它。

<aug-figure name="perspective" />

### hsv_prob

HSV 颜色抖动的概率。只有这个概率是可调的参数。抖动会在 OpenCV 的 0 到 179 色调刻度上把色调最多偏移 5，在 0 到 255 的刻度上把饱和度和明度最多偏移 30，三种偏移各自以二分之一的概率被跳过。每条读取 `hsv_prob` 的检测流水线都使用这些幅度。后三张子图是 119 次抽取中变化最明显的三次。

<aug-figure name="hsv" />

### flip_prob 和 flipud

水平翻转和垂直翻转的概率。`flip_prob` 在除 FOMO 以外的每个家族中都从 0.5 开始，FOMO 默认使用 0。`flipud` 从 0 开始。RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMO、PP-YOLOE 以及 DETR 风格的家族没有垂直翻转，会忽略它。PP-YOLOE 的配方改用随机 90 度旋转。

<aug-figure name="flip" />

### mixup_prob 和 mixup_scale

MixUp 把第二个样本混入当前样本，并保留两者的检测框。在 YOLOv9 流水线中，配对样本是一个不带 mosaic 的完整样本，混合比例在 0 到 1 之间均匀抽取，而且 `mixup_scale` 从不被读取，尽管库的 spec 把它列为受 mosaic 门控。在 YOLOX、RTMDet、PicoDet 和 FOMO 使用的 YOLOX 包装器中，以及在 YOLO-NAS 的包装器中，配对样本会先按从 `mixup_scale` 中抽取的系数缩放，然后两张图像以固定的 0.5 取平均。`mixup_prob` 的基础默认值是 1.0，YOLOv9 默认使用 0。

<aug-figure name="mixup" />

后两张子图把第二个样本混入一个 0.5× 的 mosaic。它们的图注给出混合比例 λ，即 mosaic 样本的权重。

### no_aug_epochs

`no_aug_epochs` 在各处都是 `used`，但它在各处的含义并不相同。在 mosaic 流水线上，它会在最后几轮关闭 mosaic 和 MixUp。在 DETR 风格的流水线上，它会停止光度、缩小和裁剪增强，并塑造调度的尾段。在分类流水线上，它会禁用自动增强、擦除、MixUp 和 CutMix，裁剪和翻转保留。

## DETR 风格的配方

D-FINE、Dome-DETR、DEIM、DEIMv2、RT-DETRv4 和 TinyFormer 经由同一个 torchvision 变换训练，没有 mosaic，也没有仿射变换，EdgeCrafter（EC）的检测也是如此。它的三个强操作是配方常量，而不是参数：

- `RandomPhotometricDistort` 以 0.5 的概率施加每一种颜色变化（亮度 0.875 到 1.125，对比度和饱和度 0.5 到 1.5，色调 ±0.05，以及通道打乱）。
- `RandomZoomOut` 以 0.5 的概率把图像放到一块黑色画布上，画布边长最多为图像边长的四倍，所以目标可能缩小到原尺寸的四分之一。
- `RandomIoUCrop` 以 0.8 的概率应用，裁剪出一块每条边为原边长 0.3 到 1.0 倍的区域，所选区域至少保留一部分检测框。

经过这些步骤后，任意一边不超过 1 px 的检测框会被丢弃。这三个操作都会在该家族的停止轮次停止，最晚在最后 `no_aug_epochs` 轮开始时停止。这条流水线读取的参数只有 `flip_prob` 和 `no_aug_epochs`。

<aug-figure name="detr_ops" />

一次 D-FINE 训练开头的三个样本，三个操作全部生效：

<aug-figure name="detr_samples" />

RF-DETR 有自己的流水线：先做水平翻转，然后以 `crop_resize_prob` 的概率把短边缩放到 400、500 或 600 px 并随机裁剪出 384 到 600 px 的正方形，最后直接缩放到正方形的训练分辨率，这一步是拉伸，而不是 letterbox。`crop_resize_prob` 默认为 0.5，要在 Python 中设置。

<aug-figure name="rfdetr_crop" />

## 小目标

`erasing` 不作用于检测。Random Erasing 属于分类参数包，没有任何检测流水线带有擦除步骤，所以无论它取什么值，检测器的训练都完全相同，也没有什么需要关闭的。以下这些步骤会缩小或去除小目标：

- `mosaic_scale`。在 mosaic 流水线上，目标的尺寸会乘以抽到的缩放值，所以下界决定了它最小能变成多小。在基础默认值 `(0.1, 2.0)` 下，一个 20 px 的目标可能变成 2 px。随后 YOLOv9 流水线会丢弃任意一边不超过 2 px 的检测框，而它的最终变换和 YOLOX 的一样，会丢弃任意一边不超过 1 px 的检测框。目标的像素仍留在图像中，但没有标注。下界取 0.5 时，比如 YOLOv9 的 `(0.5, 1.5)`，每个目标都会保持在推理尺寸的一半或以上。
- `degrees`、`translate`、`shear` 和 `perspective` 会把目标推到边缘，在那里它们的检测框会被截断，并经过同样的尺寸过滤。
- 在 DETR 风格的配方上，`RandomZoomOut` 可能把目标缩小到原尺寸的四分之一，`RandomIoUCrop` 可能把它们裁掉。两者都不是参数，并且都会在停止轮次停止。
- RF-DETR 的裁剪可能在裁剪边缘切断目标。`crop_resize_prob=0` 会关闭它。
- 训练分辨率会缩放每个目标：letterbox 按 `imgsz` 除以图像长边的比例缩放图像，所以在 `imgsz=1280` 时，每个目标的宽和高都是 640 时的两倍。[YOLOv9](/docs/models/yolov9) 还有一个 P2 变体，带有面向小目标的 stride-4 head。

<code-tabs name="small" />

## 分类参数包

有六个参数只作用于分类流水线。检测家族会忽略全部六个。

<code-tabs name="classify" />

`scale` 是随机裁剪的面积范围。`scale=0.5` 表示 `(0.5, 1.0)`，显式传入一对数值则同时设置两个边界。下界较小时可能把目标完全裁掉，如第三张子图所示。

<aug-figure name="cls_crop" />

`auto_augment` 接受 `"randaugment"`、`"autoaugment"`、`"augmix"` 或 `None`，并对每个样本应用一次从该策略中抽取的变换。

<aug-figure name="auto_augment" />

`erasing` 是 Random Erasing 的概率，取值在 `[0, 1)` 内。它在归一化之后运行，用零填充一个覆盖图像 2% 到 33% 的矩形，而在归一化之后，零就对应 ImageNet 的均值颜色。

<aug-figure name="erasing" />

`mixup` 和 `cutmix` 是按批生效的概率，会产生软标签。每个批次最多运行其中一个，MixUp 优先，所以两者是相加的，总和不得超过 1。在这些子图中，A 和 B 是同一张照片的两个裁剪，每个图注给出混合后的图像训练时所用的软标签。

<aug-figure name="batch_mix" />

`auto_augment`、`erasing`、`mixup` 和 `cutmix` 默认都是关闭的，所以除非你主动要求，分类训练不会有任何变化。`train()` 和 CLI 按任务分发 `mixup`：分类使用批量混合，检测则使用 `mixup_prob`。

`crop_pct=None` 保留家族的评估缩放比例。覆盖这个值会改变训练过程中所运行验证的评估裁剪，而导出仍保持家族原生的预处理。

CLI 提供 `auto_augment`、`erasing`、`cutmix`、`scale`、`crop_pct`、`fliplr` 和 `flipud`。分类的 `mixup` 表示批量混合，默认为 0.0。水平翻转默认为 0.5，垂直翻转默认为 0.0。

`no_aug_epochs` 会在最后几轮禁用自动增强、擦除、MixUp 和 CutMix，保留裁剪和翻转。分类配方默认把这个尾段设为 0。

## 家族专属参数

有些参数挂在某个家族的配置子类上，而不是基类上，所以它们只对那个家族存在，也没有 CLI 标志。

| 家族 | 参数 | 作用 |
|---|---|---|
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste` | copy-paste 实例增强的概率，仅限 `task="segment"` |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste_mode` | `"flip"` 复用同一个样本的镜像，`"mixup"` 再取一个样本 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `rot90` | 随机 90 度旋转的概率 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `max_labels` | 训练变换中每张图像的真值（ground truth）数量上限，默认 300 |
| RF-DETR | `copy_paste`、`copy_paste_mode` | 用于 `task="segment"` 的 copy-paste，仅限 `"flip"` 模式 |
| RF-DETR | `crop_resize_prob` | 原生流水线中随机裁剪缩放的概率，默认 0.5 |
| D-FINE、EdgeCrafter | `crop_resize_prob` | 随机裁剪缩放的概率，仅限 `task="segment"` |
| EdgeCrafter、YOLO-NAS | `brightness_contrast_prob`、`affine_prob` | 姿态路径的抖动概率，以及感知关键点的仿射变换概率 |

`max_labels` 是会丢数据的那个参数。超出上限的检测框会被丢弃，只发出一次警告而不报错，所以航拍这类密集图像需要把它调大。标签分配的内存取决于批次中最密集的那张图像，而不是这个上限。

不管参数怎么设置，旋转框训练都会禁用 mosaic 和 MixUp，因为针对旋转框的角点感知增强没有实现。

## 相关

- [数据增强矩阵](/docs/reference/augmentation-matrix)：每个家族中每个参数的状态、取值范围，以及每个家族的默认值。
- [超参数](/docs/train/hyperparameters)：作为调度参数的 `no_aug_epochs`，以及 `train()` 的其余参数。
- [数据集](/docs/train/datasets)：这些变换所使用的标注格式。
