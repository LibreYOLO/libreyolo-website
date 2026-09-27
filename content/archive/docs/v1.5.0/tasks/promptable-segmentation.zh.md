---
title: 可提示分割
seo_title: LibreYOLO 里的可提示分割
description: >-
  在 LibreYOLO 里把一个点、一个框或者一段文本概念变成对象掩码。通过 LibreSAM 加载 SAM、SAM 2、SAM
  3、EdgeTAM、MobileSAM 或 PicoSAM3。
lead: >-
  可提示分割把一次点击变成一张掩码：你指向一个对象，或者围着它画一个框，模型就返回它的轮廓。在 LibreYOLO
  里它不是一个单独的任务键，而是一个模型层级，通过 LibreSAM 工厂函数加载，它的结果就是普通的分割 Results。
keywords:
  - 可提示分割
  - 交互式分割
  - segment anything python
  - 点提示
  - 框提示
  - SAM python
  - 点击生成掩码
last_verified: 1.5.0
snippets:
  predict:
    - label: 点提示与框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 点是以像素为单位的 [x, y]，labels 里 1 为正，0 为负
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 多边形
        print(result.boxes.xyxy)    # 由掩码推出的紧致检测框

        # 框提示每个框给出一张掩码
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 编码一次，多次提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_image 把笨重的图像编码器跑一次并缓存下来
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: 分割一切
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 不给提示就表示在整张图上铺一个点网格；每边 32 个点的默认
        # 网格约等于 1024 次解码器前向，在 CPU 上很慢
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: 歧义掩码
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 一个点可能指袖子、衬衫，也可能指人；multimask=True 会返回
        # 这三张整体与局部的掩码，而不是最好的那一张
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## 定义

可提示分割接受一张图像加一个空间提示，返回提示所指向的那个东西的掩码。这里不做
任何分类：没有类别列表，`result.boxes` 里装的是由掩码推出的紧致检测框，而不是
自成一体的检测结果。`result.masks` 承载掩码数据，`result.masks.xy` 是它的多边形。

提示就是接口。`points` 是 `[x, y]` 像素坐标，每个对象一组，`labels` 把每个点标为
正（1，包含它）或负（0，排除它）。`bboxes` 是 `[x1, y1, x2, y2]`，每个框对应一张
掩码。点和框可以组合，这时它们按对象两两配对，长度必须一致。所有提示都不给就会走
分割一切那条路径，也就是在图上铺一个点网格。

单个点天生就是有歧义的。点在袖子上，可能指袖子、衬衫，也可能指人，所以
`multimask=True` 会为每个提示返回这三张整体与局部的掩码，而不是单独最好的那一张。
`conf` 按模型预测的 IoU 过滤，那是掩码质量分数，不是检测置信度。

LibreYOLO 没有 `promptable` 这个任务键。这一层注册为 `segment`，和实例分割用的是
同一个键。区分它们的是调用形式，这也是它有自己的工厂函数 `LibreSAM()` 的原因，它
和 `LibreYOLO()`、`LibreOpenVocab()`、`LibreVLM()` 并列。单一的 `predict(image)`
签名表达不了这些模型本来要服务的那个循环：`set_image()` 把图像编码器跑一次并缓存
嵌入向量，之后每次 `source=None` 的 `predict()` 调用只需付出提示解码的开销，
`reset_image()` 则清空缓存。图像编码器是开销的大头，每张图只跑一次，所以对同一张
图的第二次提示会完全跳过它。

## 模型

有六个家族通过别名经 `LibreSAM` 加载。

[SAM](/docs/models/sam) 是默认项，有 `base`、`large` 和 `huge` 三种尺寸，也可以
写成 `b`、`l` 和 `h`。

[SAM 2](/docs/models/sam-2)，别名为 `sam2-tiny`、`sam2-small`、`sam2-base-plus`
和 `sam2-large`。LibreYOLO 支持它的图像路径。

[SAM 3](/docs/models/sam-3)，别名为 `sam3`，是唯一接受文本概念提示的家族：
`text="yellow school bus"` 会返回每一个匹配的实例。把 `text=` 传给其他任何家族都
会抛错，错误信息里会点名 SAM 3。它的权重来自 Meta，采用自定义的 SAM 许可证
（SAM License），而不是 LibreYOLO 的 MIT 许可，而且那个仓库是受限（gated）的：
先在模型页面接受条款，并在第一次下载之前用 `hf auth login` 完成认证。部署它之前
请先读 [SAM 3](/docs/models/sam-3)。

[EdgeTAM](/docs/models/edgetam)，别名为 `edgetam`，是 SAM 2 的端侧变体。
LibreYOLO 支持它的图像路径。

[MobileSAM](/docs/models/mobilesam)，别名为 `mobilesam`，把 SAM 的 ViT-H 编码器
换成了蒸馏得到的 TinyViT 编码器。

[PicoSAM3](/docs/models/picosam3)，别名为 `picosam3`，是一个紧凑的 CNN，面向边缘端
传感器上框提示的区域。框提示就是这里的全部契约：点、文本、掩码、multimask 和分割
一切都会抛错，错误信息会指向 SAM 2 或 SAM 3。

这一层的 extra 覆盖通过 `transformers` 加载的那四个家族：

```bash
pip install "libreyolo[sam]"
```

MobileSAM 和 PicoSAM3 是 LibreYOLO 原生的移植版本，不装 `transformers` 也能跑。

## 预测

<code-tabs name="predict" />

`source` 和 `set_image()` 是二选一，不是先后两步：一次性调用就把图像传给
`predict()`，或者先调用 `set_image()`，再对每个提示调用 `predict(source=None)`。
给 `predict()` 传 `device=` 会为这次调用以及之后每一次调用移动模型，并让所有已缓存
的嵌入向量失效。

分割一切是昂贵的那种模式。`points_per_side` 默认为 32，大约是在图上跑 1024 次
解码器前向；在 CPU 上做任何交互式的事情都该把它调低。在这种模式下，不设 `conf`
会套用该家族的网格阈值，而在有提示的路径下，不设 `conf` 则保留每一张掩码。两种
模式下都可以传 `conf=0.0` 关掉过滤，并用 `max_det` 限制返回多少张掩码。

这个版本不支持掩码提示，`masks=` 会抛错，而不是被忽略。`track()` 在这一层同样会
抛错：它们都是图像分割器，所以请逐帧调用 `predict()`。输入源和结果处理见
[预测](/docs/predict)。

## 训练

这一层没有任何家族能在 LibreYOLO 内部训练。`train()` 会抛错：请在上游微调，再把
得到的权重加载进来。

## 验证

这一层没有验证器，`val()` 会抛错。可提示的掩码没有固定的类别集合可供打分，所以
常规的检测和分割指标无从下手。给一张有提示的掩码打分，意味着拿它和你自己提供的
参考掩码去比，比的是你关心的那些提示。

## 导出

导出整体不在这一层的范围内，`export()` 会抛错，只有一个例外。
[PicoSAM3](/docs/models/picosam3) 会把它那个原始的 96x96 区域 CNN 导出为 ONNX，
形式是 `roi_image -> mask_logits`；框裁剪和把掩码缩放回图像坐标的那一步留在
Python 里。其他每个家族都在 PyTorch 里通过 `predict()` 运行。库中其他地方可用的
格式见[导出](/docs/export)。
