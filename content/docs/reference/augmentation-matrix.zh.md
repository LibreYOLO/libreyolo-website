---
title: 数据增强矩阵
seo_title: LibreYOLO 各家族分别支持哪些数据增强
description: 每个数据增强参数的取值范围和 CLI 写法，每个参数在 31 个可训练家族中各自的状态，以及每个家族自带的默认值。
lead: >-
  设置一个数据增强参数，并不保证它真的会到达流水线。本页记录每个可训练家族如何对待 TrainConfig
  上的每个参数，以及每个家族的起始值，内容均由本页所记录版本的库生成。
keywords:
  - libreyolo 数据增强
  - 数据增强参数
  - 数据增强 默认值
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - 数据增强支持矩阵
  - TrainConfig 参数
last_verified: 1.8.0
verification: >-
  由 scripts/build-augmentation/collect.py 从 LibreYOLO 1.8.0 生成：参数列表、状态和说明来自
  libreyolo/data/augment/spec.py，库的测试会把它锁定到真实的流水线上；默认值来自各家族的 TrainConfig 子类；CLI
  写法来自 libreyolo train 命令。取值范围，以及 spec 与某条流水线不一致的那一处，也由同一个脚本对照库进行检查。
snippets:
  usage:
    - label: 直接查询 spec
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: 某个家族的默认值
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## 参数一览

这些是 `TrainConfig` 的字段名。`train()` 和 CLI 也接受 CLI 列中的简短写法，所以 `mosaic` 设置的是 `mosaic_prob`。在 CLI 上，`mixup` 按任务分发：对分类器是分类的批量混合，对检测器是 `mixup_prob`。`perspective` 没有 CLI 标志，要在 Python 中设置。

<aug-knobs />

基础默认值就是 `TrainConfig` 本身上的值。大多数家族会覆盖其中一部分，[各家族的默认值](#defaults-by-family)给出了一次训练实际的起始值。想看每个参数作用的图示，请见[数据增强](/docs/train/augmentations)。

方括号中的范围是区间，所以 `[0, 1)` 包含 0、不包含 1。分类参数会在构建其流水线时检查，超出范围时抛出 `ValueError`。检测参数不由 `TrainConfig` 检查，所以超出范围的值会悄无声息地照常训练。例外是抽到的缩放值小于或等于 0，以及 `perspective` 为负值，这两种情况会在样本做变换时抛出 `ValueError`。

<code-tabs name="usage" />

## 三种状态

| 状态 | 含义 |
|---|---|
| `used` | 参数会到达该家族的训练流水线，并改变样本 |
| `gated_by_mosaic` | 参数只作用于走了 mosaic 分支的样本，所以当 `mosaic_prob == 0` 时它永远不会触发 |
| `ignored` | 参数永远到不了流水线；设置它没有任何效果 |

`ignored` 是开始训练之前值得检查的那一个，因为不会有任何东西报错。当显式设置的某个训练参数正好是所选家族忽略的参数时，CLI 会发出警告；当该家族把 MixUp 门控在 mosaic 上、而 `mosaic_prob` 为零，导致 `mixup_prob > 0` 无法触发时，训练器会发出警告。

## 哪个家族认哪个参数

<aug-matrix />

所覆盖的每个家族都遵循六种流水线之一。在 YOLOX 风格的流水线中，逐样本的预处理施加 HSV 抖动和翻转，而仿射变换和 MixUp 只在 mosaic 分支内部运行。YOLO-NAS 则运行一个始终开启的逐样本仿射变换，忽略 mosaic，独立施加 MixUp，并把 `mosaic_scale` 复用为仿射变换的缩放范围。

DETR 风格的流水线是没有 mosaic 的直通变换。它的光度失真、缩小和 IoU 裁剪是配方常量，而不是可配置的参数，所以 `hsv_prob` 和几何参数永远到不了它那里。分类使用 `flip_prob` 控制水平翻转，使用 `flipud` 控制垂直翻转。语义分割的尺度抖动和 HSV 来自家族的类属性，而不是配置参数；图像复原的翻转是输入与目标耦合的操作，概率固定为 0.5。

`no_aug_epochs` 在各处都生效，只是关闭的内容不同：YOLOX 风格是 mosaic 和 MixUp；YOLO-NAS 是仿射变换和 MixUp；DETR 风格是强光度和裁剪增强，外加学习率尾段；分类是自动增强、擦除、MixUp 和 CutMix。分类的裁剪和翻转保持启用。

某个家族与同一流水线的其他家族不一致时，原因是下列之一：

| 家族 | 差异 |
|---|---|
| `rtmdet`、`picodet`、`rtdetr`、`rtdetrv2` | 忽略 `flipud`：它们的变换里没有垂直翻转 |
| `fomo` | 忽略 `perspective` 和 `flipud` |
| `ppyoloe` | 忽略 `flipud`：源配方改用随机 90 度旋转 |
| `yolo9`、`yolo9_e2e`、`yolo9_p2`、`yolo7`、`rtdetr`、`rtdetrv2` | 忽略 `mixup_scale`：它们的 MixUp 以随机比例混合两张完整的帧，两者都不缩放 |
| `gtr` | 使用 `mosaic_prob` 和 `mixup_prob`，作用于前 `mosaic_epochs` 轮；`degrees`、`translate` 和 `mosaic_scale` 决定 mosaic 之后的仿射变换 |
| `rfdetr` | 忽略 `hsv_prob`：原生流水线没有 HSV 抖动 |
| `ec` | 使用 `hsv_prob`、`degrees` 和 `translate`，仅限 `task="pose"`；detect 和 segment 使用固定的光度配方 |
| `dinov2` | 使用分类参数包，仅限 `task="classify"` |

`ec` 和 `dinov2` 是多任务家族，所以只有当该家族每一个可训练任务都忽略某个参数时，这个参数才会被标为忽略。这样，CLI 的警告就永远不会出现对一个任务是错的、对另一个任务是对的情况。

Dome-DETR 原样继承 D-FINE 的变换。它唯一不能用的是多尺度训练，关掉它的是它自己的配置，而不是数据增强 spec。

一共覆盖三十一个家族。对于不在这份列表中的家族，`ignored_aug_params` 会返回一个空集合，所以不会为它发出任何警告。

## 各家族的默认值

每个检测家族的配置所用的起始值。空单元格表示该家族忽略这个参数，所以它的默认值不改变任何东西。

<aug-defaults />

有两个规律比任何单个取值都重要。RT-DETR、RT-DETRv2、PicoDet 和 FOMO 以 `mosaic_prob=0` 起步，所以在这些家族上，矩阵中标为受门控的参数在开启 mosaic 之前不起任何作用。YOLOv9 起步时关闭旋转、错切和 MixUp，并使用 `mosaic_scale=(0.5, 1.5)`，而基础默认值是 `(0.1, 2.0)`。

EdgeCrafter 只在 `task="pose"` 时读取 `hsv_prob`、`degrees` 和 `translate`，所以它那一行显示的是这三个参数在姿态任务下的默认值。

分类家族从基础默认值起步：`flip_prob=0.5`、`flipud=0.0`、`scale=(0.5, 1.0)`、家族自己的 `crop_pct`，以及关闭的 `auto_augment`、`erasing`、`mixup` 和 `cutmix`。

## 家族专属参数

有些家族把数据增强参数挂在自己的 `TrainConfig` 子类上，而不是基类上。CLI 不暴露这些参数；请通过 Python API 设置。

| 家族 | 参数 | 含义 |
|---|---|---|
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste` | copy-paste 实例增强的概率，仅限 `task="segment"` |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste_mode` | copy-paste 的来源：`flip` 镜像同一个样本，`mixup` 使用第二个样本 |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `rot90` | 随机 90 度旋转的概率 |
| `rfdetr` | `copy_paste` | `task="segment"` 的 copy-paste 概率，仅限 `flip` 模式 |
| `rfdetr` | `copy_paste_mode` | `task="segment"` 的 copy-paste 来源模式 |
| `rfdetr` | `crop_resize_prob` | 原生流水线中随机裁剪缩放的概率 |
| `dfine` | `crop_resize_prob` | 随机裁剪缩放的概率，`task="segment"` |
| `ec` | `crop_resize_prob` | 随机裁剪缩放的概率，`task="segment"` |
| `ec`、`yolonas` | `brightness_contrast_prob` | 亮度和对比度抖动的概率，`task="pose"` |
| `ec`、`yolonas` | `affine_prob` | 感知关键点的仿射变换概率，`task="pose"` |

`rot90` 在 `yolo9` 上作用于 detect 和 OBB。

## 查询 spec

| 辅助函数 | 返回 |
|---|---|
| `aug_support(family)` | 参数到 `Support` 的对照表，未知家族返回 `None` |
| `ignored_aug_params(family)` | 该家族忽略的参数名集合；未知家族返回空集合 |
| `uses_mosaic_gating(family)` | 该家族的 MixUp 是否只在 mosaic 样本上触发 |
| `display_name(family)` | 警告中使用的、面向用户的家族名 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUp 永远无法触发时的警告文本，否则为 `None` |

`Support` 是一个由 `status` 和 `note` 组成的具名元组，其中 note 说明某个参数为什么在该家族上被忽略或受门控。在 1.8.0 中，spec 仍把 `mixup_scale` 列为在经由 YOLOv9 mosaic 包装器训练的家族上受门控，所以 `aug_support` 会在这些家族上报告它受门控，尽管上面经过流水线核对的矩阵把它标为忽略。

## mosaic 门控

对 YOLOX 风格的家族来说，`mixup_prob=0.5` 配上 `mosaic_prob=0` 会完全关掉 MixUp，因为 MixUp 只作用于 mosaic 样本。在训练后期关掉 mosaic 时，很容易出现这个组合。训练器会记录一条点名该家族的警告，`mixup_gating_warning` 就是它背后的纯函数。
