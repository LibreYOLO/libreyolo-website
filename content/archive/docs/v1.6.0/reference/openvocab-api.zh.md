---
title: 开放词汇 API
seo_title: LibreOpenVocab API：别名与参数
description: >-
  LibreOpenVocab 工厂、它的四个家族和全部别名、set_classes、各家族的 conf 默认值，以及 text_threshold 和
  iou 的规则。
lead: >-
  LibreOpenVocab 是文本条件检测器的工厂。类别列表是一段提示词，而不是固定的 head，所以词汇表由 set_classes
  设定，模型据此返回针对它的普通检测 Results。
keywords:
  - LibreOpenVocab
  - 开放词汇目标检测
  - Grounding DINO 文本提示检测
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes 设置类别
last_verified: 1.5.0
verification: >-
  别名读取自 libreyolo/models/openvocab/__init__.py；仓库、尺寸和阈值来自
  grounding_dino.py、owlv2.py、omdet_turbo.py 和 ov_deim.py；调用规则来自
  libreyolo/models/openvocab/base.py，全部基于 v1.5.0。设计意图来自
  docs/adr/0008-open-vocab-detector-contract.md。
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## 安装

这一层需要 `openvocab` extra。

<code-tabs name="install" />

## 工厂

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` 是一个别名，不是路径。查找之前下划线会折叠成连字符，所以 CLI 清单打印出
的那些带家族前缀的名字，比如 `omdet_turbo-t` 和 `grounding_dino-t`，照原样就能加
载。未知的别名会抛出 `ValueError`，并列出每一个已知的别名。

构造函数接收 `size`、`nb_classes=80`、`names=None`、`device="auto"`、`task=None`
和 `text_threshold=None`。传 `names` 等同于加载完立刻调用一次 `set_classes`。把
`text_threshold` 传给不支持它的家族会抛出 `TypeError`。

<code-tabs name="usage" />

## 家族与别名

| 家族 | 别名 | 尺寸 | 权重 |
|---|---|---|---|
| Grounding DINO | `grounding-dino`、`groundingdino`、`grounding-dino-tiny`、`groundingdino-tiny`、`grounding-dino-t`、`groundingdino-t`、`grounding-dino-base`、`groundingdino-base`、`grounding-dino-b`、`groundingdino-b` | `t`、`b` | `LibreYOLO/LibreGroundingDINOt`、`LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`、`owl-v2`、`owlv2-base`、`owl-v2-base`、`owlv2-b16`、`owl-v2-b16`、`owlv2-large`、`owl-v2-large`、`owlv2-l14`、`owl-v2-l14` | `b16`、`l14` | `LibreYOLO/LibreOWLv2b16`、`LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`、`omdet`、`omdetturbo`、`omdet-turbo-tiny`、`omdet-turbo-swin-tiny`、`omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`、`ovdeim`、`ov-deim-s`、`ovdeim-s`、`ov-deim-m`、`ovdeim-m`、`ov-deim-l`、`ovdeim-l` | `s`、`m`、`l` | `LibreYOLO/LibreOVDEIMs`、`LibreYOLO/LibreOVDEIMm`、`LibreYOLO/LibreOVDEIMl` |

默认别名是 `grounding-dino-tiny`。

`LibreGroundingDINO`、`LibreOWLv2` 和 `LibreOMDetTurbo` 在包级别导出，可以直接用
`size=` 构造。OV-DEIM 则通过上面那些工厂别名访问。

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

为之后每一次 `predict()` 调用设定词汇表，并返回模型本身，所以调用可以链式书写。
列表必须非空，元素必须都是字符串，并且在不区分大小写比较时必须互不重复；空白标签
会被拒绝。直接传一个字符串会抛出 `TypeError`，因为它会被枚举成一堆单字符类别。

调用之后，`model.names` 会按给定的顺序把 `0..N-1` 映射到各个标签，
`model.nb_classes` 就是 `N`。

## 调用参数

这一层复用标准的预测接口，只有三处不同。

`conf` 的默认值取各家族自己的取值，而不是共用的 0.25：

| 家族 | 默认 conf | 抑制 |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | 它自己的后处理，阈值 0.5，采纳 `iou=` |
| OV-DEIM | 0.25 | 一对一匹配加 top-K 选取，不做抑制 |

`iou=` 只对会跑抑制的家族有意义。OMDet-Turbo 把阈值当作一个参数接收，`iou=` 没有
设置时默认为 0.5。另外三个什么都不抑制，所以在那里传 `iou=` 会发出一条警告，并且
被忽略。

`text_threshold=` 只有 Grounding DINO 有，在那里默认为 0.25。它可以在构造时传入
作为一个长期生效的值，也可以每次调用时传。按次传的值不能和 `stream=True` 一起
用，因为流式的结果是惰性生成的；这种情况请改在构造函数上设置。其他每个家族碰到它
都会抛出 `TypeError`。

`imgsz=` 会抛出 `ValueError`：这一层的缩放由预处理流水线负责。`augment=True` 同样
会抛错，因为测试时增强（TTA）不在这里的范围内。各家族的输入尺寸只作记录，仅供参
考：Grounding DINO 800、OWLv2 960 和 1008、OMDet-Turbo 640、OV-DEIM 640。

## 不支持

`train()`、`val()`、`track()` 和 `export()` 都会抛出 `NotImplementedError`。请在
上游做微调，再把得到的权重加载进来；跟踪则用逐帧调用 `predict()` 代替。验证需要一
个专门的验证器，因为共用的检测验证器是拿图像张量去调用模型的，而这一层要的是文本
条件的输入。
