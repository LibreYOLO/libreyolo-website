---
title: 数据增强矩阵
seo_title: LibreYOLO 各家族分别支持哪些数据增强参数
description: 按家族划分的数据增强参数支持情况：TrainConfig 上的十六个参数、三种状态、六种流水线原型，以及某个家族会静默忽略的参数。
lead: >-
  设置一个数据增强参数，并不保证它真的会到达流水线。本页记录每个可训练家族如何对待 TrainConfig
  上的每个参数，依据的是库自带的那张声明式表格——它是唯一的事实来源。
keywords:
  - libreyolo 数据增强
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - 数据增强支持矩阵
  - TrainConfig 参数
last_verified: 1.5.0
verification: >-
  参数列表、状态、原型、各家族的差异以及辅助函数，均读自 v1.5.0 的 libreyolo/data/augment/spec.py。那张表格由
  tests/unit/test_augment_spec.py 锁定到真实的流水线上。
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
source_hash: d2e1b9f5c81072e1
---

## 参数一览

这些是 `TrainConfig` 的字段名，不是 CLI 的写法。CLI 把自己的别名映射到它们上面，
所以 `--mosaic` 设置的是 `mosaic_prob`。

| 参数 | 含义 |
|---|---|
| `mosaic_prob` | 构建一个 4 图 mosaic 样本的概率 |
| `mixup_prob` | 混入第二个样本的概率 |
| `hsv_prob` | HSV 颜色抖动的概率 |
| `flip_prob` | 水平翻转概率 |
| `degrees` | 仿射变换的随机旋转范围，单位为度 |
| `translate` | 仿射变换的随机平移比例 |
| `mosaic_scale` | 仿射变换的随机缩放范围 |
| `mixup_scale` | 施加在 MixUp 配对图像上的抖动缩放范围 |
| `shear` | 仿射变换的随机错切范围，单位为度 |
| `perspective` | 仿射变换的投影变换强度 |
| `flipud` | 垂直翻转概率 |
| `no_aug_epochs` | 训练末尾关闭强增强的轮数 |
| `auto_augment` | 分类的 AutoAugment 策略：randaugment、autoaugment 或 augmix |
| `erasing` | 分类的 RandomErasing 概率 |
| `mixup` | 分类的 batch-MixUp 概率，带软标签 |
| `cutmix` | 分类的 batch-CutMix 概率，带软标签 |

最后四个是分类那一组。检测家族会忽略它们。`mixup` 是只在 API 上存在的参数：CLI 的
`--mixup` 是检测端 `mixup_prob` 的别名。

<code-tabs name="usage" />

## 三种状态

| 状态 | 含义 |
|---|---|
| `used` | 参数会到达该家族的训练流水线，并改变样本 |
| `gated_by_mosaic` | 参数只作用于走了 mosaic 分支的样本，所以当 `mosaic_prob == 0` 时它永远不会触发 |
| `ignored` | 参数永远到不了流水线；设置它没有任何效果 |

`ignored` 是开始训练之前值得检查的那一个，因为不会有任何东西报错。当显式设置的某个训练
参数正好是所选家族忽略的参数时，CLI 会发出警告；当 `mixup_prob > 0` 却因为该家族把
MixUp 门控在 mosaic 上、而 `mosaic_prob` 为零而无法触发时，训练器会发出警告。

## 流水线原型

本页覆盖的每个家族都遵循六种流水线之一，个别家族另有偏差，列在下面。

| 参数 | YOLOX 式 | YOLO-NAS | DETR 式 | 分类 | 语义分割 | 复原 |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | ignored | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | ignored | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

在 YOLOX 式流水线里，逐样本的预处理会施加 HSV 抖动和翻转，而仿射变换和 MixUp 只在
mosaic 分支内部运行。YOLO-NAS 则是跑一个始终开启的逐样本仿射，忽略 mosaic，并独立
施加 MixUp，把 `mosaic_scale` 复用为仿射的缩放范围。

DETR 式流水线是一个直通变换，没有 mosaic。它的光度扰动、zoom-out 和 IoU 裁剪是配方
里的常量，而不是可配置的参数，这就是 `hsv_prob` 和那些几何参数永远到不了它那里的
原因。分类流水线用的是 ImageFolder 变换，它的水平翻转是固定的 0.5，而不是
`flip_prob`。语义分割的尺度抖动和 HSV 来自家族的类属性，而不是配置参数；复原的翻转
则是输入和目标耦合在一起的操作，概率固定为 0.5。

`no_aug_epochs` 在所有地方都被遵守，只是它关掉的东西各不相同：YOLOX 式关掉 mosaic
和 MixUp，YOLO-NAS 关掉仿射和 MixUp，DETR 式关掉强光度增强和裁剪增强以及学习率的
尾段，其余的则关掉调度器的尾段。

## 各原型下的家族

| 原型 | 家族 |
|---|---|
| YOLOX 式 | `yolox`、`yolo7`、`yolo9`、`yolo9_e2e`、`yolo9_p2`、`rtmdet`、`picodet`、`rtdetr`、`rtdetrv2`、`fomo` |
| YOLO-NAS | `yolonas` |
| DETR 式 | `dfine`、`domedetr`、`deim`、`deimv2`、`rtdetrv4`、`rfdetr`、`ec`、`dinov2` |
| 分类 | `resnet`、`convnext`、`mobilenetv4`、`efficientnetv2` |
| 语义分割 | `segformer` |
| 复原 | `nafnet` |

一共覆盖二十五个家族。不在这份列表里的家族会返回一个空的忽略集合，所以不会为它发出
任何警告。

## 偏差

| 家族 | 与其原型的差异 |
|---|---|
| `rtmdet` | 忽略 `flipud`：它的变换里没有垂直翻转 |
| `picodet` | 忽略 `flipud` |
| `rtdetr` | 忽略 `flipud` |
| `rtdetrv2` | 忽略 `flipud` |
| `fomo` | 忽略 `perspective` 和 `flipud` |
| `ec` | 会使用 `hsv_prob`、`degrees` 和 `translate`，但仅限 `task="pose"`；detect 和 segment 使用固定的光度配方 |
| `dinov2` | 会使用分类那一组，但仅限 `task="classify"` |

`ec` 和 `dinov2` 是多任务家族，所以只有当该家族每一个可训练任务都忽略某个参数时，这
个参数才会被标为 ignored。这样一来，CLI 的警告就不会出现对一个任务说错、对另一个任
务说对的情况。

Dome-DETR 原样继承 D-FINE 的变换。它唯一吃不下的是多尺度训练，而关掉它的是它自己的
配置文件，不是数据增强 spec。

## 家族专有参数

有些家族把数据增强参数挂在自己的 `TrainConfig` 子类上，而不是基类上。CLI 不暴露这些
参数；请通过 Python API 设置。

| 家族 | 参数 | 含义 |
|---|---|---|
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste` | copy-paste 实例增强的概率，仅限 `task="segment"` |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste_mode` | copy-paste 的来源：`flip` 镜像同一个样本，`mixup` 使用第二个样本 |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `rot90` | 随机 90 度旋转的概率 |
| `rfdetr` | `copy_paste` | `task="segment"` 的 copy-paste 概率，仅支持 `flip` 模式 |
| `rfdetr` | `copy_paste_mode` | `task="segment"` 的 copy-paste 来源模式 |
| `rfdetr` | `crop_resize_prob` | 原生流水线里随机裁剪缩放的概率 |
| `dfine` | `crop_resize_prob` | 随机裁剪缩放的概率，`task="segment"` |
| `ec` | `crop_resize_prob` | 随机裁剪缩放的概率，`task="segment"` |
| `ec`、`yolonas` | `brightness_contrast_prob` | 亮度和对比度抖动的概率，`task="pose"` |
| `ec`、`yolonas` | `affine_prob` | 关键点感知的仿射概率，`task="pose"` |

在 `yolo9` 上，`rot90` 作用于 detect 和 OBB。

## 查询 spec

| 辅助函数 | 返回 |
|---|---|
| `aug_support(family)` | 参数到 `Support` 的对照表，未知家族返回 `None` |
| `ignored_aug_params(family)` | 该家族忽略的参数名集合；未知家族返回空集合 |
| `uses_mosaic_gating(family)` | 该家族的 MixUp 是否只在 mosaic 样本上触发 |
| `display_name(family)` | 警告里使用的、面向人的家族名 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUp 永远无法触发时的警告文本，否则为 `None` |

`Support` 是一个由 `status` 和 `note` 组成的具名元组，其中 note 说明某个参数为什么在
该家族上被忽略或被门控。

## mosaic 门控

对 YOLOX 式家族来说，`mixup_prob=0.5` 配上 `mosaic_prob=0` 会完全关掉 MixUp，因为
MixUp 只作用于 mosaic 样本。在训练后期关掉 mosaic 时，很容易凑出这个组合。训练器会
记一条点名该家族的警告，`mixup_gating_warning` 就是它背后的纯函数。
