---
title: 核心概念
seo_title: LibreYOLO 核心概念
description: LibreYOLO 里任务、模型家族、尺寸和检查点文件名之间是怎么对应的，以及每个支持层级各自承诺了什么。
lead: 四个概念就能描述 LibreYOLO 里的每一个模型：它执行的任务、它所属的家族、它在家族内的尺寸，以及这个家族所处的支持层级。检查点文件名编码了前三个。
keywords:
  - libreyolo 核心概念
  - libreyolo 任务类型
  - libreyolo 模型家族
  - libreyolo 检查点命名
  - libreyolo 支持层级
last_verified: 1.5.0
meta:
  - label: 文件名格式
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: 标准任务名
    value: 17
  - label: 支持层级
    value: 'Flagship, Core, Supported, Inference only, Museum, Sibling tier'
snippets:
  inspect:
    - label: 列出家族
      language: bash
      code: |
        # 每个已注册家族的任务、尺寸和输入分辨率
        libreyolo models
    - label: 单个模型
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: 指定任务
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 别名在 API 边界处归一化："keypoints" 解析为
        # "pose"，"det" 解析为 "detect"，"semantic-segmentation" 解析为 "semantic"
        model = LibreYOLO("LibreYOLO9t.pt", task="det")
        print(model.task)
source_hash: 23d045463a6a8411
---

## 任务

任务就是模型返回的东西。LibreYOLO 有十七个标准任务名，每一个都对应 `Results`
对象上承载其输出的那个字段。

| 任务 | 返回内容 |
|---|---|
| `detect` | 带类别和置信度的轴对齐检测框 |
| `segment` | 逐实例掩码，每检测到一个目标就有一张掩码 |
| `semantic` | 每个像素一个类别标签，不区分实例 |
| `panoptic` | 每个像素一个互不重叠的标签，把可数的物体（things）与无定形的背景（stuff）合并在一起 |
| `pose` | 逐实例关键点，每行与检测框一一对应 |
| `classify` | 整张图片在一个标签集上的概率 |
| `obb` | 旋转框，带一个旋转角 |
| `point` | 每个检测输出一个图像坐标，而不是一个框 |
| `depth` | 一张稠密的相对逆深度图 |
| `normal` | 一个稠密的单位向量表面法线场 |
| `edge` | 一张稠密的边缘概率图 |
| `restore` | 一张复原后的 RGB 图像，用于去模糊、去噪或超分辨率 |
| `matte` | 一张从 0 到 1 的软前景图，用于去除背景 |
| `ocr` | 带转写文本的文字四边形，按阅读顺序排列 |
| `embed` | 一个 L2 归一化向量，它的点积衡量吻合程度 |
| `gaze` | 每张检测到的人脸一个视线方向 |
| `mesh` | 每个检测到的人一个带姿态的 3D 人体 |

这些就是出现在检查点（checkpoint）元数据和文件名里的名字。在任何可以传入任务的
地方，也接受大家熟悉的别名，并且在其他一切发生之前先做归一化：`detection` 和
`det` 变成 `detect`，`keypoints` 变成 `pose`，`cls` 变成 `classify`，`deblur`、
`denoise` 和 `super-resolution` 都变成 `restore`，`face-recognition` 和 `reid`
变成 `embed`。无法识别的名字会直接报错，而不是悄悄退回默认值。

`segment`、`semantic` 和 `panoptic` 是三个不同的任务，不是同一件事的三种说法。
实例掩码、逐像素标签和 thing 加 stuff 的合并图有不同的真值（ground truth）、
不同的指标和不同的结果字段。

## 模型家族

一个家族就是一条架构脉络，有自己的加载、预处理和后处理代码。每个家族都会声明一个
`FAMILY` 标识符，比如 `yolo9`、`rfdetr` 或 `dfine`，声明它支持的任务，以及它发布的
每个尺寸对应的输入分辨率。

`LibreYOLO()` 是一个工厂，而不是一个类。给它一个路径，它会加载这个文件，从检查点
元数据里识别出家族，识别不出来就从张量的键本身去识别，然后返回该家族模型的一个
实例。这就是为什么换一个检测器只是改一行代码：拿回来的对象暴露同样的 `predict`、
`train`、`val` 和 `export` 接口，也返回同样的 `Results` 类型。

<code-tabs name="inspect" />

支持多个任务的家族通常为每个任务单独发布一个检查点，而且每个任务的尺寸集合往往
还不一样；也有少数家族让两个运行时任务共用同一个文件。无论哪种情况，支持的任务都是
一个固定的列表，请求列表之外的任务会报错，并在消息里给出支持的列表，而不是加载一个
差不多的东西。

完整列表，连同每个家族的基准测试和已发布的权重，见[所有模型](/docs/models)。

## 尺寸

尺寸是家族内部的一个变体，写成一个小写代号，直接接在家族前缀后面。常见的字母是 `n`
表示 nano、`t` 表示 tiny、`s` 表示 small、`m` 表示 medium、`l` 表示 large、`x`
表示 xlarge，但代号是各家族自己定的，有几个家族用的完全是别的东西：以骨干命名的
代号，比如 `r50` 或 `r101`，此时尺寸是 ResNet 的深度；复合缩放的代号，比如 `b0`
到 `b3`；或者一个用来标识那唯一一个已发布检查点的名字。YOLOv9 在其他家族用 `l`
的地方用 `c` 表示 compact。

尺寸同时也决定了输入分辨率，对于有多个任务的家族，分辨率还可能因任务而异。这两者
都从家族里读取，绝不靠假设；`libreyolo models` 会把它们打印出来。

## 检查点文件名

每一个已发布的权重文件都遵循同一套格式：

```text
Libre<FAMILY><size>[-<task>].pt
```

家族前缀是每个家族固定的字符串，尺寸是小写并且不加分隔符直接接上，任务后缀以
连字符开头。检测不带后缀，沿用 YOLO 检查点一直以来的惯例，所以 `LibreYOLO9t.pt`
是一个检测器，而 `LibreRFDETRn-seg.pt` 是同一个家族的分割模型。

| 任务 | 后缀 |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

如果一个家族没有任何不带后缀的任务，它可以强制要求带上后缀，这样一个不带后缀的
名字就不会被当作它的合法检查点。如果一个家族发布了在默认数据集之外的数据集上训练的
权重，会把数据集名字再作为一个后缀接上去，而这个变体也是文件下载来源的仓库名的
一部分。

有三类家族不在这套格式之内。可提示（promptable）的分割家族、视觉语言家族和开放词汇
检测器没有注册进检查点工厂，也不产出 `Libre<FAMILY><size>.pt` 文件。它们的前缀指向的
是下载下来的 Hugging Face 快照或者一个可提示的检查点，那里上游的品牌大小写是特意
保留的。

## 任务是怎么确定的

当有多个信号都可能给出任务名时，它们会按固定顺序依次查询，第一个存在的胜出：你传入的
`task` 参数，然后是检查点元数据里记录的任务，然后是文件名里的任务后缀，最后是家族的
默认任务。得到的结果在构建模型之前会拿去和家族支持的任务做校验，所以不匹配会在加载
时就失败，而不是之后产出错误的输出。

## 支持层级

每个家族都只归属于一个层级。层级说的是工程投入，而不是精度：它告诉你一个新功能会
先落在哪里，以及哪些东西的 CI 会一直保持绿色。

| 层级 | 含义 |
|---|---|
| Flagship | 功能在这里最先设计，并完整地在 GPU 上验证 |
| Core | 核心的可训练检测器。功能在同一个发布波次里紧跟 Flagship |
| Supported | 起支撑作用的可训练家族。在 CI 里保持绿色，功能视情况落地 |
| Inference only | 预测、验证和导出。训练相关的功能不适用 |
| Museum | 一件冻结的展品。只修 bug |
| Sibling tier | 一个独立的产品面，有自己的工厂和契约 |

每个模型页面都在页头标出所属家族的层级。两个 Flagship 家族分别是 CNN 检测器里的
[YOLOv9](/docs/models/yolov9) 和 transformer 检测器里的
[RF-DETR](/docs/models/rf-detr)；除非你有别的理由，否则就从它们开始。

Inference only 说的是缺了什么，缺的是 LibreYOLO 里的训练循环。预测、验证，以及在
家族支持的情况下导出，都是能用的。对这样的家族调用 `train()` 会抛出
`NotImplementedError`，并说明原因。
