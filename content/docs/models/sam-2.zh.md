---
title: SAM 2
families: [sam2]
seo_title: "SAM 2：LibreYOLO 里的可提示图像分割"
description: "在 LibreYOLO 里用 SAM 2 做基于点提示和框提示的分割。安装 tiny、small、base-plus 和 large 检查点并做预测，Apache-2.0。"
lead: "SAM 2 在 SAM 的基础上加了一套为视频设计的流式记忆（streaming memory）架构，把一次点击或框选变成一张物体掩码。LibreYOLO 通过专门的 LibreSAM 工厂函数支持它的图像分割路径，与 LibreYOLO() 检测器工厂函数分开。"
keywords: [SAM 2, Segment Anything, "sam2 python", "可提示分割", "交互式分割", "点提示分割", "sam2 掩码", Hiera]
last_verified: "1.5.0"
snippets:
  predict:
    - label: 点提示与框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # 尺寸别名："sam2-tiny"、"sam2-small"、"sam2-base-plus"、
        # "sam2-large"（也有简写 "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l"）
        model = LibreSAM("sam2-large")

        # 点提示：像素坐标下的 [x, y]，标签 1 = 前景
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 每张掩码一个多边形
        print(result.boxes.xyxy)    # 由掩码推出的紧致检测框

        # 用框提示代替点提示
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # 完全不给提示就分割整张图（一个简化版的自动掩码生成器，
        # 不是那个穷举式的参考实现）
        result = model.predict(SAMPLE_IMAGE)
    - label: 编码一次，多次提示
      language: python
      code: |
        from libreyolo import LibreSAM2, SAMPLE_IMAGE

        # 家族专用的类只接受尺寸，不带 "sam2-" 前缀
        model = LibreSAM2("large")

        # 图像编码器是开销最大的部分，set_image() 只跑一次，
        # 之后每次 predict() 调用都会复用缓存下来的嵌入向量
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
---

## 安装

SAM 2 需要 `sam` 这个额外依赖，它会带上 `transformers` 和 `timm`。

```bash
pip install "libreyolo[sam]"
```

## 预测

`LibreSAM(...)`（或者家族专用的 `LibreSAM2(...)`）是和 `LibreYOLO(...)`
分开的入口：它返回的是一个可提示的分割器，而不是检测器，因为在这里，没有空间提示的
一次前向传播毫无意义。这个家族没有 `libreyolo predict` 这条 CLI 命令；请用
Python API。只支持图像分割；SAM 2 的视频记忆跟踪不在这里的范围内。

<code-tabs name="predict" />

点提示接受表示单个物体的 `[x, y]`、表示多个物体的 `[[x, y], ...]`，或者 numpy
数组；`labels` 把每个点标记为 `1`（前景）或 `0`（背景），默认全部是前景。框提示接受
`[x1, y1, x2, y2]` 或者一组框，每个框一张掩码。两种提示都不给，就会用一张稠密网格去
提示、并保留其中置信且互不重叠的掩码，从而分割整张图；这个「分割一切」模式相对参考
实现的自动掩码生成器做了简化，在拥挤场景下可能分割不足，所以真正精确的路径还是给一个
点提示或框提示。`conf` 按预测的掩码质量（IoU）过滤，而不是按检测置信度：传 `0.0`
可以保留每一个候选。`multimask=True` 会为每个提示返回 SAM 那三张整体与部件的歧义
掩码，而不是只返回最好的一张。`device=` 会移动模型，如果有一个 `set_image()` 会话
正在进行，还会一并移动它缓存的嵌入向量。每张掩码的类别 id 都是 `0`，名字是
`"object"`，因为可提示掩码没有固定的类别集合。这个家族的 `train()`、`val()`、
`export()` 和 `track()` 都会抛出 `NotImplementedError`：LibreYOLO 在这里支持的
就是图像推理。输入源类型见[预测](/docs/predict)。

## 变体

四种 Hiera 骨干尺寸：tiny、small、base-plus 和 large，输入分辨率都一样。这个家族
目前还没有发布精度或延迟的基准测试，所以选尺寸就是在编码器体量和掩码质量之间直接
权衡：tiny 编码最快，large 最重。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可

<provenance-box></provenance-box>

## 引用

<citation-block />
