---
title: 数据增强
seo_title: LibreYOLO 中的训练数据增强
description: TrainConfig 上的数据增强参数、它们背后的四种流水线形态，以及一张按家族说明哪些参数被使用、被门控、被忽略的表。
lead: >-
  数据增强由 TrainConfig 上的参数配置，但每个模型家族跑的是自己的训练流水线，而一条没有 mosaic 分支的流水线会直接忽略
  mosaic_prob，而不是拿别的东西去近似它。
keywords:
  - yolo 数据增强
  - mosaic 增强
  - mixup
  - hsv 数据增强
  - 随机仿射变换
  - copy paste 数据增强
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
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
  support:
    - label: 读某个家族的支持表
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
source_hash: 47461cd13aab580c
---

## 设置这些参数

数据增强参数就是普通的 `train()` 参数。

<code-tabs name="train" />

其中两个在 CLI 上有更短的写法：`mosaic` 对应 `mosaic_prob`，`mixup` 对应
`mixup_prob`。其余参数在两边的写法完全一致。

## 三种状态，而不是两种

一个参数是否真的起作用，取决于家族。库里维护了一张声明式的表，每一项都是三种状态
之一。

`used` 表示这个参数会进入流水线并改变样本。`ignored` 表示它根本进不了流水线，设了
也没用。`gated_by_mosaic` 表示它只作用于走了 mosaic 分支的样本，所以在
`mosaic_prob=0` 时，即使它已经接好了线也永远不会触发。

第三种状态才是让人意外的那个。在 YOLOX 风格的流水线上，仿射变换跑在 mosaic 画布
上，MixUp 混合的也是 mosaic 样本，所以 `mosaic_prob=0` 会一次性悄悄关掉
`degrees`、`translate`、`shear`、`perspective`、`mosaic_scale`、`mixup_prob` 和
`mixup_scale`。训练器只针对 MixUp 这种情况打了一条警告：

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI 也会对被忽略的参数发出警告，而且只列出你真正写了的那几个：

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## 四种流水线形态

各个家族归到四条训练流水线里，而流水线几乎决定了所有答案。

YOLOX 风格的 mosaic 流水线按样本做 HSV 抖动和翻转，然后在 mosaic 分支里跑仿射变换
和 MixUp。它覆盖 YOLOX、YOLOv7、YOLOv9 及其 E2E 和 P2 变体、RTMDet、PicoDet、
RT-DETR、RT-DETRv2 和 FOMO。

DETR 风格的直通流水线既没有 mosaic 也没有仿射变换。它的光度失真、zoom-out 和 IoU
裁剪都是配方里的常量，而不是配置参数，所以只有 `flip_prob` 和 `no_aug_epochs` 是
活的。它覆盖 D-FINE、Dome-DETR、DEIM、DEIMv2、RT-DETRv4、EC，以及改了一处的
RF-DETR。

分类的 ImageFolder 流水线忽略每一个检测参数。它的水平翻转是固定的 0.5，`flip_prob`
够不着。它有自己的一套参数，见下文。

YOLO-NAS 自成一种形态：完全没有 mosaic，逐样本的仿射变换始终开着，MixUp 是独立应用
的，而不是被门控的。它的 `mosaic_scale` 值被复用为仿射变换的缩放范围。

SegFormer 和 NAFNet 各自跑一条任务专属的流水线，其随机性写死在家族里，不可配置。对
SegFormer 来说，活的参数是类属性 `semantic_scale_jitter` 和 `semantic_hsv_prob`，
而不是 `mosaic_scale` 和 `hsv_prob`。NAFNet 的裁剪和翻转是输入与目标耦合的操作，概
率固定为 0.5。

## 哪个家族认哪个参数

下面这张表就是随库分发的 spec，位于 `libreyolo/data/augment/spec.py`，库自己的测试
会拿它和真实的流水线接线做断言。请直接去那里读，而不是从架构去推。

<code-tabs name="support" />

按流水线汇总，只看基础参数：

| 参数 | YOLOX 风格 | YOLO-NAS | DETR 风格 | 分类 |
|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored |
| `mixup_prob` | 受 mosaic 门控 | used | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored |
| `flip_prob` | used | used | used | ignored |
| `flipud` | used | used | ignored | ignored |
| `degrees` | 受 mosaic 门控 | used | ignored | ignored |
| `translate` | 受 mosaic 门控 | used | ignored | ignored |
| `shear` | 受 mosaic 门控 | used | ignored | ignored |
| `perspective` | 受 mosaic 门控 | used | ignored | ignored |
| `mosaic_scale` | 受 mosaic 门控 | used | ignored | ignored |
| `mixup_scale` | 受 mosaic 门控 | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used |

这些列内部还有一些例外，全都是收窄的：

- RTMDet、PicoDet、RT-DETR、RT-DETRv2 和 FOMO 没有垂直翻转，所以 `flipud` 被忽略。
  FOMO 的 mosaic 包装器构建时也不带 perspective。
- RF-DETR 原生的流水线没有 HSV 抖动，所以在 DETR 风格那一列之外，`hsv_prob` 也被
  忽略。
- EC 认 `hsv_prob`、`degrees` 和 `translate`，但仅限 `task="pose"`，因为只有它那个
  感知关键点的变换会读这些值。它的 detect 和 segment 路径用的是固定的光度配方。
- DINOv2 的 detect 和 semantic 任务遵循 DETR 风格那一列，并为 `task="classify"` 加
  上分类参数包。

`no_aug_epochs` 在所有家族上都是 `used`，但它在各处的含义并不相同。在 mosaic 流水
线上，它会在最后几轮关掉 mosaic 和 MixUp。在 DETR 风格的流水线上，它会停掉光度、
zoom-out 和裁剪这几种增强，并改变调度的尾部。在分类和语义分割流水线上，它只改变尾
部。

## 分类参数包

有四个参数只驱动分类流水线，别的什么都不驱动。检测家族对这四个一律忽略。

<code-tabs name="classify" />

`auto_augment` 接受 `"randaugment"`、`"autoaugment"`、`"augmix"` 或 `None`。
`erasing` 是 RandomErasing 的概率。`mixup` 和 `cutmix` 是按批生效的概率，产生软标
签；每批最多跑一个，MixUp 优先，所以两者是相加的，总和不应超过 1。

四个默认都是关的，所以除非你主动要求，分类训练不会有任何变化。

有一处命名冲突值得直说：在 CLI 上，`mixup` 是检测那个 `mixup_prob` 的别名。分类的
`mixup` 字段没有自己的 CLI 写法，只能在 Python 里通过 `model.train(mixup=...)` 才
够得着。

## 家族专属参数

有些参数挂在某个家族的配置子类上，而不是基类上，所以它们只对那个家族存在，也没有
CLI flag。

| 家族 | 参数 | 作用 |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | copy-paste 实例增强的概率，仅 `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` 复用同一张样本的镜像，`"mixup"` 会再取一张样本 |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | 随机 90 度旋转的概率 |
| YOLOv9 | `max_labels` | 训练变换里每张图的真值（ground truth）上限，默认 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | 用于 `task="segment"` 的 copy-paste，仅 `"flip"` 模式 |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | 随机裁剪缩放的概率 |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | pose 路径的抖动概率和感知关键点的仿射变换概率 |

`max_labels` 是那个会悄悄丢数据的参数。超过上限的检测框会被直接丢掉，不报错，所以
航拍这类密集图像需要把它调大。

不管参数怎么设，旋转框训练都会禁用 mosaic 和 MixUp，因为针对旋转框、感知角点的增强
还没有实现。

## 相关

- [超参数](/docs/train/hyperparameters)：`no_aug_epochs` 作为调度参数，以及 `train()`
  的其余部分。
- [数据集](/docs/train/datasets)：这些变换消费的标注格式。
