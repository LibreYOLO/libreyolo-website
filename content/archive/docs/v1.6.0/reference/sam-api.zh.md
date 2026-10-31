---
title: 可提示分割 API
seo_title: LibreSAM API：提示、别名与签名
description: LibreSAM 工厂函数、尺寸别名，点、框和概念文本三种提示类型，一次编码的 set_image 生命周期，以及这一层不支持的东西。
lead: >-
  LibreSAM 是可提示分割的工厂函数。一次前向传播需要在调用时给出针对该图像的提示，所以这一层有自己的 predict
  接口，而不是走无提示的推理运行器。
keywords:
  - LibreSAM
  - sam 可提示分割
  - sam 点提示
  - sam 框提示
  - set_image
  - 分割一切
  - libreyolo sam 扩展
last_verified: 1.5.0
verification: >-
  工厂别名、尺寸和仓库读自
  libreyolo/models/sam/model.py、sam2.py、edgetam.py、sam3.py、libreyolo/models/mobilesam/model.py
  和 libreyolo/models/picosam3/model.py。提示契约与默认值读自
  libreyolo/models/sam/base.py。设计意图来自 docs/adr/0007-libresam-contract.md，全部基于
  v1.5.0。
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: 点提示与框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 一次编码，多次提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## 安装

这一层需要 `sam` 扩展。

<code-tabs name="install" />

## 工厂函数

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` 是尺寸别名，不是路径。`**kwargs` 会传到家族的构造函数，它接受 `device`
和 `multimask`。未知别名会抛出 `ValueError`，错误信息里会列出所有已知别名。

<code-tabs name="usage" />

## 别名

| 家族 | 别名 | 尺寸 | 权重 |
|---|---|---|---|
| SAM-1 | `base`、`large`、`huge`、`b`、`l`、`h`、`sam-base`、`sam-large`、`sam-huge`、`sam_b`、`sam_l`、`sam_h` | `base`、`large`、`huge` | `facebook/sam-vit-base`、`-large`、`-huge` |
| SAM-2 | `sam2-tiny`、`sam2-small`、`sam2-base-plus`、`sam2-baseplus`、`sam2-large`，以及简写形式 `sam2-t`、`sam2-s`、`sam2-bp`、`sam2-l`、`sam2_t`、`sam2_s`、`sam2_bp`、`sam2_l` | `tiny`、`small`、`base-plus`、`large` | `LibreYOLO/LibreSAM2tiny`、`-small`、`-base-plus`、`-large` |
| EdgeTAM | `edgetam`、`edge-tam`、`edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`、`sam-3`、`sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`、`mobilesam-tiny`、`mobilesam_t`、`mobile-sam`、`mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`、`picosam3-pico`、`picosam3_pico`、`pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

默认是 `base`。SAM-1、SAM-2、EdgeTAM 和 MobileSAM 跑在名义上 1024 像素的画布
上，SAM 3 是 1008，PicoSAM3 是 96。

SAM 3 的权重是受限的（gated）。它们从 `facebook/sam3` 下载，采用 Meta 自定的
SAM License，既不是 MIT 也不是 Apache-2.0，LibreYOLO 不做再分发。加载前先在仓库
页面接受条款，并通过 Hugging Face 认证；加载器会先打印这条提示。

家族类也一并导出，所以 `LibreSAM1`、`LibreSAM2`、`LibreSAM3`、`LibreEdgeTAM`、
`LibreMobileSAM` 和 `LibrePicoSAM3` 可以用 `size=` 直接构造。

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `source` | `None` | 要分割的图像；`None` 表示复用 `set_image()` 缓存的图像 |
| `points` | `None` | 以像素坐标给出的点提示 |
| `bboxes` | `None` | 形如 `[x1, y1, x2, y2]` 的框提示，也可以给一组框，每个框出一张掩码 |
| `labels` | `None` | 点标签，`1` 为正、`0` 为负，形状与 `points` 对应；省略时全部按正处理 |
| `masks` | `None` | 保留参数；传入会抛出 `NotImplementedError` |
| `text` | `None` | 概念提示；仅 SAM 3 支持 |
| `conf` | `None` | 预测掩码 IoU 的下限 |
| `multimask` | `None` | 每个提示返回全部歧义掩码；默认取构造时的设置 |
| `max_det` | `300` | 返回掩码数量的上限 |
| `device` | `None` | 为这次以及之后的调用移动模型，并让缓存的嵌入向量失效 |
| `color_format` | `"auto"` | 内存中数组的色彩格式提示 |
| `points_per_side` | `None` | 分割一切的网格密度；默认 32 |

返回的是普通的 `Results`，带 `masks`，外加由这些掩码推出的紧致 `boxes`，类别 `0`
命名为 `"object"`。

## 提示的形状

`points` 接受这些嵌套形式：单个对象用 `[x, y]`，N 个对象用 `[[x, y], ...]`，按
对象分组的点用 `[[[x, y], ...], ...]`。凡是能用列表的地方都能用 Numpy 数组。
坐标就是源图像上的普通像素值。

不给任何空间提示就会跑分割一切：一个基于网格的自动掩码生成器，带预测 IoU 阈值和
框 IoU 去重。`points_per_side` 默认 32，大约要跑 1024 次解码器前向，在 CPU 上很
慢；交互使用时把它调小。这个生成器省掉了稳定性分数过滤、多裁剪和掩码 IoU 去重，
所以它只是有提示路径的近似，而不是与之等价。

## 置信度

`conf` 按预测掩码 IoU 过滤，那是掩码质量分数，不是检测置信度。`None` 在有提示的
路径下保留每一张掩码，在分割一切下则套用该家族的网格阈值。`0.0` 在两种模式下都
关闭过滤。

在 SAM 3 的文本路径上，`conf` 换成 Promptable Concept Segmentation 的检测分数。
那里的 `None` 表示标准的 0.3 阈值，`0.0` 则保留全部候选。

## 文本提示

`text=` 只有 SAM 3 支持；所有走空间提示的家族碰到它都会抛出
`NotImplementedError`。文本与点、框互斥。返回的 `names` 把类别 `0` 映射到请求的
概念。`source=None` 的文本调用会重新编码缓存的图像，因为跟踪器和概念编码器不共用
一份缓存。

关键字 `exemplars=` 是为将来的图像范例扩展保留的，目前没有实现。

## 一次编码的生命周期

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` 把笨重的图像编码器跑一次，并缓存嵌入向量，所以之后每次
`source=None` 的 `predict()` 都很便宜。两个方法都返回模型本身，调用可以链式写。
给 `predict` 传 `device=` 会移动模型，并让缓存失效。

## PicoSAM3

PicoSAM3 只接受 `bboxes=`。点、文本、掩码、multimask 和分割一切的提示都会抛错。
框会外扩 10%，再送进一个 96 像素的 ROI 网络；PicoSAM3 是这一层里唯一能导出的
家族，而且只能导出 ONNX。

## 不支持的功能

`train()`、`val()` 和 `track()` 在这一层的每个家族上都会抛出
`NotImplementedError`。可提示的掩码没有固定的类别集合可供打分，所以 mAP 在这里
没有意义。`export()` 在 SAM-1、SAM-2、SAM 3、EdgeTAM 和 MobileSAM 上都会抛错。

SAM-2、SAM 3 和 EdgeTAM 的视频与 memory 路径不在这个版本的范围内，SAM 3 的图像
范例和掩码提示同样不在。
