---
title: 导入已有权重
seo_title: 在 LibreYOLO 中加载上游权重
description: 把 LibreYOLO 指向上游项目的检查点。自动转换会在加载时重新封装它，并保留它的类别数和类别名。
lead: LibreYOLO 的模型家族是从上游项目移植过来的，所以这些项目发布的检查点几乎已经可以直接加载。它们缺的是元数据。自动转换在加载时把这部分补上。
keywords:
  - libreyolo 转换权重
  - 加载上游检查点
  - libreyolo 迁移权重
  - pth 转 libreyolo
  - 权重自动转换
last_verified: 1.5.0
meta:
  - label: 入口
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: 写在源文件旁边，命名为
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: 转换脚本
    value: 仓库中的 weights/
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 把路径换成你已有的某个检查点，识别得出的上游布局会被即时转换、
        # 写到源文件旁边，然后再加载
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # 类别数和类别名来自张量以及文件自带的元数据，
        # 所以微调过的模型保留的是它自己的标签集，而不是 COCO 的
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 检查结果
      language: bash
      code: |
        # 转换出来的文件满足和官方发布文件相同的 schema
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

本页讲的是来自其他项目的检查点（checkpoint）。如果你是把自己的代码从旧版
LibreYOLO 迁过来，见[升级到 1.5.0](/docs/upgrade)。

## 加载外来文件时会发生什么

`LibreYOLO()` 会先用受限的、仅加载权重的路径读取任何权重文件。如果读出来的东西
带有完整的 LibreYOLO 元数据，就直接使用。如果没有，这个文件会在尝试其他任何做法
之前先交给自动转换器。如果受限加载直接失败——检查点里 pickle 了第三方对象时就会
这样——则改用一个会中和这些对象的加载器再走一遍自动转换。

自动转换做四件事。它把张量字典从上游项目所用的那种封装布局里拆出来。它逐个询问
已注册的家族是否认得拆出来的键，并在上游命名和 LibreYOLO 的移植版不一致时重映射
名称。它把胜出的那个封装成满足元数据 schema v1.0 的检查点，其中的尺寸、任务和
类别数都从张量本身读出。然后它把结果写到源文件旁边，再加载这个新文件。

<code-tabs name="convert" />

转换过程不是静默的。每转换一个文件，日志里都会记下家族、源文件名、输出文件名和
得到的类别数，这样一次运行的日志就准确记录了到底加载了什么。

## 它能拆开的布局

上游检查点会把权重嵌套在少数几个约定俗成的位置，转换器按顺序逐个尝试，直到某个
位置里确实装着张量：`ema.module` 下的 EMA 块或扁平的 `ema`、去掉 `module.` 前缀的
`ema_state_dict`，然后是 `params_ema`、`params`、`ema_net`、`net`、`model`、
`state_dict`，最后是对象本身。逐个尝试而不是只看第一个，意味着一个只装着计数器的
`ema` 块不会挡住它下面真正的权重。

外层前缀也会被剥掉：分布式训练留下的 `module.`、编译后模型留下的 `_orig_mod.`，
以及某些再分发版本额外套上的一层 `model.model.` 嵌套。

## 它读什么，从哪里读

尺寸、任务和类别数来自张量，而不是文件名，所以微调过的检查点转换出来带的是它自己
的类别数，而不是这个架构的默认值。类别名在检查点自带的元数据里有的时候就取自那里，
名称放在 `args` 或 `hyper_parameters` 块里的时候就取自那里，并且会按检测到的类别数
截断，这样一个保留了基础标签集的微调结果就不会带上它的 head 已经没有的索引。

密集任务是显式处理的，而不是给它编造标签。深度检查点得到一个名为 `depth` 的类别，
图像修复检查点得到一个名为 `image` 的类别。姿态检查点必须能给出关键点数量，要么
来自张量，要么来自家族；如果两者都给不出，转换会被拒绝，而不是写出一个不完整的
文件。

RF-DETR 有自己的识别器，因为判断尺寸需要看整个检查点，也因为它的 head 有 91 个
输出，而 LibreYOLO 用的是 80 类的 COCO 约定。当一个检查点正好带 80 个类别名、或
声明类别数为 80、或把 COCO 写成自己的数据集、或根本不带类别与数据集元数据时，它会
被归一到 80 类。而真正的 90 类模型——由它的类别名、明确的非 80 类别数或非 COCO 的
数据集线索识别出来——则原样保留。

## 转换后的文件写到哪里

输出写在源文件旁边，并以源文件命名：

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

于是一个存成 `upstream-checkpoint.pth` 的 tiny 尺寸 YOLOv9 检测器会变成
`upstream-checkpoint-LibreYOLO9t.pt`。按源文件而不是按家族命名，意味着同一个目录下
同家族同尺寸的两次微调不会互相覆盖，也都不会和官方检查点撞名。这个文件在每次加载时
都会重写，所以它相对源文件永远不会过期。如果目录是只读的，转换后的文件会改写到一个
新建的私有临时目录里，日志会说明写到了哪里。

从那以后它就是一个普通的 LibreYOLO 检查点：它走元数据路径加载，`libreyolo metadata`
也会报告它有效。

## 需要人工介入的情况

有两个家族不走通用识别器。视线估计家族被直接排除：它只做推理，而且它发布的权重带有
再分发限制。RF-DETR 被排除，是因为它有上面说的那个专用识别器，改由那个识别器处理。

未经处理的上游 PIDNet 检查点会被拒绝，报错会指向
`weights/convert_pidnet_weights.py`。那个脚本会写入检查点需要的 Cityscapes 语义
分割元数据。

D-FINE 和 DEIM 共用同一套架构键名，所以光靠张量分不出它们。当两者都认领同一个文件、
而候选里又没有带区分标记的同门家族时，由文件名决定：形如 `dfine_hgnetv2_n_coco.pth`
或 `deim_hgnetv2_n_coco.pth` 的名字就能定下来，而什么都说明不了的名字会带着这条
解释被拒绝，不会去猜。直接实例化 `LibreDFINE` 或 `LibreDEIM` 同样能解决。

当多个家族都合理地认领同一个文件时，子类胜过它所细化的基类，其余则由注册表
（registry）顺序决定，因为这个顺序编码了每个家族的判定条件有多具体。文件名只在
D-FINE 和 DEIM 打平时才会参考，所以文件名永远不可能让一个宽泛的匹配压过一个精确的
匹配。

## 转换脚本

仓库在 `weights/` 下带有按家族划分的转换脚本，还有一批处理重复琐碎工作的共用辅助
代码。运行时路径拒绝掉的文件、想提前而不是在加载时生成检查点、以及那些元数据必须
手动提供而无法从张量推断的家族，走的都是这条路。

这些脚本属于仓库，而不属于安装后的包，所以要用就得先克隆：

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

每个脚本写出的检查点都满足 schema v1.0，这和自动转换达到的标准一样，也和官方发布的
权重达到的标准一样。关于这套 schema 包含什么，见[检查点与权重](/docs/weights)。
