---
title: SAM 3
families: [sam3]
seo_title: "SAM 3：LibreYOLO 里的可提示分割与概念分割"
description: "在 LibreYOLO 里用 SAM 3 做点提示、框提示和文本概念分割。安装并用 large 检查点跑预测，这份权重受 Meta 的 SAM 许可证限制。"
lead: "SAM 3 在常规的点提示和框提示之上，给 SAM 加了文本概念提示，于是像「黄色校车」这样的短语会返回每一个匹配的实例。LibreYOLO 通过一个专门的 LibreSAM 工厂函数支持它的图像路径，与 LibreYOLO() 检测器工厂函数分开。"
keywords: [SAM 3, Segment Anything, Meta AI, "可提示分割", "概念分割", "用文字分割图片", "sam3 文本提示", "交互式分割 python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: 点提示与框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "sam3" 是唯一的尺寸（"large"），别名有 "sam3"、"sam-3"、"sam3-large"
        model = LibreSAM("sam3")

        # 点提示：[x, y] 为像素坐标，label 1 = 前景
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 每个掩码一个多边形
        print(result.boxes.xyxy)    # 由掩码推出的紧致检测框

        # 用框提示代替点提示
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 文本（概念）提示
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # 找出匹配该短语的每一个实例，而不只是一个物体
        # text= 与 points、bboxes、labels 和 masks 互斥
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # 每个实例的 PCS 检测分数
    - label: 编码一次，多次提示
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # 图像编码器是最耗时的部分，set_image() 只跑一次；
        # 之后每次 predict() 调用都复用缓存的嵌入向量。
        # text= 调用会在内部重新编码，因为跟踪器和
        # 概念分割编码器不共享缓存
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## 安装

SAM 3 需要 `sam` 这个 extra，它会一并装上 `transformers` 和 `timm`。

```bash
pip install "libreyolo[sam]"
```

权重是受限（gated）的：先访问
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)，接受
Meta 的 SAM 许可证（SAM License），然后在第一次下载之前运行 `hf auth login`
（或设置 `HF_TOKEN`）。LibreYOLO 第一次下载这个家族时会记录一条许可证提示。

## 预测

`LibreSAM(...)`（或者家族专用的 `LibreSAM3(...)`）是和 `LibreYOLO(...)`
分开的入口：它返回的是一个可提示（promptable）分割器，而不是检测器，因为在这里
没有提示的一次前向传播毫无意义。这个家族没有 `libreyolo predict` CLI 命令；
请用 Python API。只支持图像推理；SAM 3 的视频模型不在这里的范围内。

<code-tabs name="predict" />

点提示和框提示这条路径和 SAM 家族的其他成员一致：点提示接受 `[x, y]` 表示单个
物体，或 `[[x, y], ...]` 表示多个，`labels` 把每个点标为 `1`（前景）或 `0`
（背景），框提示接受 `[x1, y1, x2, y2]` 或一组框。这条路径上的 `conf` 按预测出的
掩码质量（IoU）过滤，而不是检测置信度。

`text=` 这条路径是 SAM 3 新加的：一个概念字符串会通过可提示概念分割
（Promptable Concept Segmentation）返回图中每一个匹配的实例，并且不能和点、框、
labels 或掩码组合使用。那里的 `conf` 是 PCS 检测分数，而不是掩码 IoU；保持默认值
就会用模型自己的 0.3 阈值，`conf=0.0` 则保留每一个候选。返回的 `names` 把类别 id
`0` 映射到请求的那个概念字符串，因为可提示的掩码本来没有固定的类别集合。
`device=` 会移动模型，如果有正在进行的 `set_image()` 会话，也会一并移动它缓存的
嵌入向量。对这个家族来说，`train()`、`val()`、`export()` 和 `track()` 都会抛出
`NotImplementedError`：SAM 3 在 LibreYOLO 里只能做预测，视频跟踪也不在范围内。
输入源类型见[预测](/docs/predict)。

## 变体

只有一个尺寸 large，输入固定为 1008 px。SAM 3.1 不支持：它的实现带有一个自定义
许可证，无法收录进这个 MIT 许可的仓库，而且 LibreYOLO 依赖的 Transformers 版本
还加载不了它的检查点（checkpoint）格式。

## 许可

<provenance-box>

LibreYOLO 不托管自己的那份 SAM 3 权重，也不转发它们。`LibreSAM("sam3")` 直接从
Meta 在 Hugging Face 上受限的 `facebook/sam3` 仓库下载，而那个仓库要求在第一次
下载之前接受 Meta 的 SAM 许可证并完成认证。

</provenance-box>

## 引用

<citation-block />
