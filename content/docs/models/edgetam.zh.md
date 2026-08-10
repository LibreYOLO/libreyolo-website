---
title: EdgeTAM
families: [edgetam]
seo_title: "EdgeTAM：LibreYOLO 里的端侧可提示分割"
description: "在 LibreYOLO 里用 EdgeTAM 做点提示和框提示分割，为端侧速度而生。安装并用这个 Apache-2.0 许可的检查点跑预测。"
lead: "EdgeTAM 是 SAM 2 的端侧变体，为移动端推理速度而构建，同时保留同样的点提示和框提示工作流。LibreYOLO 通过一个专门的 LibreSAM 工厂函数支持它的图像分割路径，与 LibreYOLO() 检测器工厂函数分开。"
keywords: [EdgeTAM, SAM 2, "可提示分割", "交互式分割 python", "点一下分割物体", "端侧分割", "sam2 轻量版", Meta Reality Labs]
last_verified: "1.5.0"
snippets:
  predict:
    - label: 点提示与框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # EdgeTAM 只有一个尺寸 "edge"，别名："edgetam"、"edge-tam"、
        # "edgetam-edge"
        model = LibreSAM("edgetam")

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
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # 图像编码器是最耗时的部分，set_image() 只跑一次；
        # 之后每次 predict() 调用都复用缓存的嵌入向量
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## 安装

EdgeTAM 需要 `sam` 这个 extra，它会一并装上 `transformers` 和 `timm`。

```bash
pip install "libreyolo[sam]"
```

## 预测

`LibreSAM(...)`（或者家族专用的 `LibreEdgeTAM(...)`）是和 `LibreYOLO(...)`
分开的入口：它返回的是一个可提示（promptable）分割器，而不是检测器，因为在这里
没有空间提示的一次前向传播毫无意义。这个家族没有 `libreyolo predict` CLI 命令；
请用 Python API。只支持图像分割；EdgeTAM 的视频跟踪不在这里的范围内。

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
LibreYOLO 在这里支持的就是图像推理。输入源类型见[预测](/docs/predict)。

## 变体

只有一个尺寸 edge，输入分辨率固定，所以在 SAM 这一档里选择这个家族是个硬件决策，
而不是尺寸决策：EdgeTAM 存在的意义就是受限的端侧推理。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可

<provenance-box></provenance-box>

## 引用

<citation-block />
