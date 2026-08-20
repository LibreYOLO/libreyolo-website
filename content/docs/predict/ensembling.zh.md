---
title: 检测器集成
seo_title: LibreYOLO 中的检测器集成
description: 在同一张图像上运行多个检测器，用加权检测框融合或 NMS 把它们的检测框融合起来，包括类别列表不同的模型。
lead: >-
  LibreEnsemble 在同一张解码后的图像上运行两个或更多检测器，把它们的检测框融合成一个 Results
  对象。各成员保留自己的权重、阈值、设备和类别列表。
keywords:
  - 模型集成 目标检测
  - weighted boxes fusion
  - wbf python
  - 多个检测器结果融合
  - 检测框融合
  - LibreEnsemble
  - 集成推理 python
  - min_votes
last_verified: 1.5.0
verification: >-
  构造函数与调用签名、默认值、验证错误、类别空间统一、票数统计以及返回的 Results 读自
  libreyolo/ensemble/model.py。融合算法及其参数来自 libreyolo/ops/fusion.py。设计意图来自
  docs/adr/0004-model-ensembling.md。使用方式与 tests/unit/test_ensemble.py 和
  tests/unit/test_ops_fusion.py 交叉核对。
snippets:
  basic:
    - label: 两个检测器，融合结果
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # 成员可以是检查点路径，也可以是已加载的模型
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: 权重与投票要求
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # 按惯例，与验证 mAP 成正比
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # 只保留两个成员都找到的检测框
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: 按成员设置阈值
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # 标量对每个成员都生效，列表按成员逐一读取
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: 接入 LibreYOLO 没有加载的检测器
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # 返回 (boxes, scores, labels)：xyxy 为原图像素坐标
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: 与单个模型相同的输入源
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # 把 clip.mp4 换成磁盘上的视频文件
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 6dcd2f84ec6f3f65
---

## 什么是集成

`LibreEnsemble` 接受两个或更多检测器，让每个都在同一张图像上运行，再把它们的检测框
融合成一个 `Results`。它是预测时的构造：没有任何东西需要训练，各成员仍然是独立的
模型，可以各自单独验证和导出。

检测是它唯一支持的任务。任务不是检测的成员会在构造时抛出 `ValueError`，并指出该成员
的索引和它的任务。

这两个名字都是惰性导入的，用不到就不产生任何开销：

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## 构建一个集成

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` 是一个包含两个或更多元素的序列。`str` 或 `Path` 条目会通过 `LibreYOLO()`
加载，其他任何东西都必须是可调用的，并暴露一个 `names` 字典。少于两个会抛出
`ValueError`，而直接传入一个裸字符串会抛出 `TypeError`，而不是去遍历它的字符。

`weights` 默认为 `None`，也就是均匀加权。给出的权重必须每个成员一个，且严格为正，
所以权重为零会抛出异常，而不是悄悄丢掉一个成员。文档约定是把它们设成与各成员的
验证 mAP 成正比。

`fusion_iou` 默认为 `0.55`，它是把来自不同成员的检测框聚成一簇所用的 IoU。它和每次
调用传入的 `iou` 是不同的阈值，后者是各成员自己的 NMS 设置。

`min_votes` 默认为 `1`，意思是任何单个成员都能让一个检测框留下来。调高它，就只保留
被那么多个不同成员确认过的簇。它必须是不大于成员数量的正整数，并且会按类别被限制到
真正认识该类别的成员数量，这样只有一个成员训练过的类别就不会被悄悄抹掉。

## 融合方法

有三种可以按名字指定，也接受传入一个可调用对象。

| `fusion` | 行为 |
|---|---|
| `"wbf"` | 加权检测框融合（weighted boxes fusion），顺序执行，忠于论文 [1]。默认值 |
| `"wbf_seeded"` | 单遍加权检测框融合，由类别感知的 NMS 挑选簇的种子 |
| `"nms"` | 把每个成员的检测框拼接起来，然后做类别感知的 NMS |

[1] Roman Solovyev, Weimin Wang, Tatiana Gabruseva, ["Weighted boxes fusion:
Ensembling boxes from different object detection models"](https://arxiv.org/abs/1910.13302),
arXiv:1910.13302.

加权检测框融合按置信度加权平均一簇框的坐标，得到的框没有任何单个成员提出过。两个
加权变体在簇没有歧义时结果一致，在重叠簇连成链时可能略有差别。`"nms"` 是挑出一个
幸存者而不是取平均，所以幸存下来的框保留原始分数，权重只影响哪个框胜出。因为它是
做选择而不是聚簇，所以它无法统计票数：把 `fusion="nms"` 和大于 `1` 的 `min_votes`
一起使用会抛出 `ValueError`。

加权检测框融合会按支持某一簇的成员权重占比来重新缩放这一簇的分数。两个成员权重相等
时，只有其中一个找到的框只保留一半分数：`0.9` 变成 `0.45`。因此融合后的置信度可能
低于各成员运行时使用的 `conf`，所以要按融合后的分数过滤，不要以为成员的阈值仍然
成立。

## 类别列表不同的成员

各成员不必共用同一个类别列表。它们的标签空间按名称取并集，每个成员会拿到一张查找表，
把自己的类别 id 重新映射到并集里。`ensemble.names` 就是这个并集，也是返回的
`Results` 所携带的。

检测框只会在同一个类别名称内部融合。只有一个成员认识的类别会不经融合直接通过，也
不会因此受到惩罚：分数缩放用的是按类别计算的分母，所以只有一家认识的类别保留它的
分数。

部分重叠会记录一条警告，指出哪些类别不是每个成员都有的。这条警告值得仔细读，因为
一个类别名是 `class_0` 这类占位符的检查点（checkpoint），构建出的并集与其他每个成员
都不相交，于是根本不会发生跨成员的融合。

成员返回自己 `names` 之外的类别 id 会抛出 `RuntimeError`。

## 外部检测器

<code-tabs name="external" />

`ExternalDetector(fn, names)` 会包装任何接受一张 PIL 图像并返回
`(boxes, scores, labels)` 的可调用对象，其中检测框是原图像素坐标下的 xyxy。它会校验
参数个数、框的形状、长度是否一致，以及每个类别 id 是否都出现在 `names` 里，并且由它
自己施加 `conf` 阈值。

LibreYOLO 没有加载的检测器就是这样参与融合的。

## 调用

<code-tabs name="sources" />

调用签名与单个模型的一致，也接受同样的输入源：图像、文件夹、列表、视频、屏幕捕获、
摄像头和网络流。实时源需要 `stream=True`，原因和在别处一样。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `conf` | `0.25` | 按成员生效，标量会广播，也可以每个成员一个 |
| `iou` | `0.45` | 各成员自己的 NMS 阈值，不是融合阈值 |
| `imgsz` | `None` | `list` 按成员逐一读取，`int` 或元组会广播 |
| `device` | `None` | 标量或每个成员一个，因此成员可以待在不同设备上 |
| `classes` | `None` | 按并集类别 id 过滤融合后的结果 |
| `max_det` | `300` | 作用于融合后的结果 |

因为对 `imgsz` 来说 `list` 表示按成员指定，`imgsz=[480, 640]` 就是第一个成员用 480、
第二个成员用 640，而 `imgsz=(480, 640)` 是对所有成员统一的一个矩形尺寸。这个区别很
容易踩坑。

不管你要求多少，成员都会以至少 300 的 `max_det` 被调用，这样每个成员都跑得宽松些，
由集成在最后统一裁剪一次。

图像只解码一次，同一个对象交给每个成员。`batch` 为了保持一致而被接受，但会被忽略；
图像是顺序处理的。

## 返回什么

一个普通的 `Results`，和单个模型返回的类型相同，只是 `names` 设为并集类别空间。
[处理结果](/docs/predict/results)上的一切都原样适用。

唯一的区别是 `result.speed`，集成确实会填充它。它的键是 `member_0`、`member_1`
这样的，外加 `fusion`，单位是毫秒。这是库里唯一会填上 `speed` 的地方。

含有非有限检测框或分数的行会在融合前被丢弃。当成员待在不同设备上时，融合在第一个
返回了结果的成员所在的设备上进行。

## 集成做不到的事

`val()` 和 `export()` 都会抛出 `NotImplementedError`，并把你指向成员：逐个验证、
逐个导出。根本没有 `train` 方法，所以调用它会抛出 `AttributeError`。

半精度不在集成这一层处理。`half=True` 走的还是和别处一样、带警告的空操作路径；请在
每个成员上配置精度。

集成没有命令行接口。它是一个 Python API。
