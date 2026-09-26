---
title: SAM
families:
  - sam
seo_title: SAM（Segment Anything）：在 LibreYOLO 里预测掩码
description: >-
  在 LibreYOLO 里用 SAM 做点提示和框提示的可提示分割。安装并用 Apache-2.0 许可的 base、large 和 huge
  检查点跑预测。
lead: >-
  SAM（Segment Anything）把一次点击的点或框变成物体掩码。LibreYOLO 通过一个专门的 LibreSAM 工厂函数加载它，与
  LibreYOLO() 检测器工厂函数分开，因为一个可提示的模型需要不同的调用形式。
keywords:
  - SAM
  - Segment Anything
  - 可提示分割
  - 交互式分割 python
  - 点一下分割物体
  - sam 分割一切
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: 点提示与框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "base" 会在首次使用时自动下载 facebook/sam-vit-base
        # 其他尺寸："large"、"huge"（也可以写 "b"/"l"/"h"）
        model = LibreSAM("base")

        # 点提示：[x, y] 为像素坐标，label 1 = 前景
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 每个掩码一个多边形
        print(result.boxes.xyxy)    # 由掩码推出的紧致检测框

        # 用框提示代替点提示
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # 完全不给提示则分割整张图（一个简化版的自动掩码生成器，
        # 而不是详尽的参考实现）
        result = model.predict(SAMPLE_IMAGE)
    - label: 编码一次，多次提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 图像编码器是最耗时的部分，set_image() 只跑一次；
        # 之后每次 predict() 调用都复用缓存的嵌入向量
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: f8904d241ef8a929
---

## 安装

SAM 需要 `sam` 这个 extra，它会一并装上 `transformers` 和 `timm`。

```bash
pip install "libreyolo[sam]"
```

## 预测

`LibreSAM(...)` 是和 `LibreYOLO(...)` 分开的入口：它返回的是一个可提示
（promptable）分割器，而不是检测器，因为在这里没有空间提示的一次前向传播毫无
意义。这个家族没有 `libreyolo predict` CLI 命令；请用 Python API。

<code-tabs name="predict" />

点提示接受 `[x, y]` 表示单个物体，`[[x, y], ...]` 表示多个，也接受 numpy 数组；
`labels` 把每个点标为 `1`（前景）或 `0`（背景），默认全部为前景。框提示接受
`[x1, y1, x2, y2]` 或一组框，每个框对应一个掩码。两种提示都不给时，它会用一个稠密
网格去提示，并保留其中置信度高、互不重叠的掩码，以此分割整张图；这种「分割一切」
模式相对参考实现的自动掩码生成器做了简化，在拥挤场景里可能欠分割，所以真正给一个
点提示或框提示才是精确的那条路。`conf` 按预测出的掩码质量（IoU）过滤，而不是检测
置信度：传 `0.0` 就保留每一个候选。`multimask=True` 会为每个提示返回 SAM 那三个
整体与部分的歧义掩码，而不是只返回最好的那一个。`device=` 会移动模型，如果有正在
进行的 `set_image()` 会话，也会一并移动它缓存的嵌入向量。每个掩码的类别 id 都是
`0`，名字是 `"object"`，因为可提示的掩码没有固定的类别集合。对这个家族来说，
`train()`、`val()`、`export()` 和 `track()` 都会抛出 `NotImplementedError`：
SAM 在 LibreYOLO 里只支持预测，视频跟踪不在范围内。输入源类型见[预测](/docs/predict)。

## 变体

三种 ViT 图像编码器尺寸：base、large 和 huge，输入都固定在 1024 px。这个家族目前
还没有发布精度或延迟的基准测试，所以选尺寸就是在编码器的重量和掩码质量之间直接
权衡：base 编码最快，huge 最重。

## 许可

<provenance-box>

LibreYOLO 不会自己托管一份 SAM-1 权重的副本。`LibreSAM("base")`、`"large"` 和
`"huge"` 直接从 Meta 自己在 Hugging Face 上的 `facebook/sam-vit-base`、
`facebook/sam-vit-large` 和 `facebook/sam-vit-huge` 仓库下载，这几个仓库各自
独立于 LibreYOLO 标注为 Apache-2.0。

</provenance-box>

## 引用

<citation-block />
