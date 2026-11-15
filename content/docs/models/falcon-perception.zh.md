---
title: Falcon Perception
families:
  - falconperception
seo_title: Falcon Perception Python 用法：文本查询驱动的检测与分割
description: >-
  在 LibreYOLO 里运行 TII 的 Falcon Perception：找出与文本查询匹配的每个实例并返回检测框，600m 上还能同时返回掩码。采用
  Apache-2.0 许可。
lead: >-
  Falcon Perception 是技术创新研究院（TII）推出的早期融合
  transformer，它读取一张图像和一条自然语言查询，返回所有匹配的实例。LibreYOLO 内置了它的推理网络，两个尺寸都支持开放词汇检测，600m
  还支持实例分割。
keywords:
  - Falcon Perception
  - TII
  - 开放词汇检测
  - 指代表达
  - 文本提示分割
  - grounding 定位
  - 早期融合
  - falcon perception python
  - LibreVLM
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # 每个词表条目是一条查询；实例拿到该条目的类别 id
        model = LibreVLM("falcon-perception", names=["building", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.xyxy)
    - label: 分割
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("falcon-perception", names=["the titanium building"])
        result = model.set_task("segment").predict(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)          # 每个实例一个掩码
    - label: 300m
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # 较小的尺寸没有掩码 head：只能检测
        model = LibreVLM("falcon-perception-300m", names=["building"])
        result = model.predict(SAMPLE_IMAGE)
source_hash: 51373e45e0b278ec
---

## 安装

Falcon Perception 属于 LibreYOLO 的 VLM 层级，需要 `vlm` extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 TII 自己的仓库下载，即 `tiiuae/Falcon-Perception`（600m）和
`tiiuae/Falcon-Perception-300M`，固定到某个 revision。网络内置在 LibreYOLO 里；
不运行任何远程代码。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 或 `LibreFalconPerception` 类加载。必须提供 `names=`（或
`set_classes()`）。每个词表条目对应一条文本查询和一次生成，所以开销随条目数增长，
为某个条目找到的每个实例都拿到该条目的类别 id。查询可以是类别名，也可以是
`"the dog on the left"` 这样的短语。

对每个实例，模型输出一个框中心、一个框尺寸和一个掩码 query。`detect` 返回
`Results.boxes`；`segment`（仅 600m）还返回 `Results.masks`，由掩码 query 与经过
AnyUp 上采样的图像特征做点积得到。置信度恒为 1.0，也没有 NMS：上游的坐标去重在
解码过程中进行（`coord_dedup_threshold`，默认 0.01）。每条查询的生成预算是 2048 个
新 token（`max_new_tokens`）。

## 变体

| 名称 | 尺寸 | 任务 |
| --- | --- | --- |
| `falcon-perception`、`falcon-perception-600m` | 600m（默认） | detect、segment |
| `falcon-perception-300m` | 300m | detect |

两者都以 1024 px 运行。只有 MPS 可用时，这个家族会在 CPU 上运行，因为它的 RoPE 表是
complex64，而 MPS 没有实现这种类型。在 CPU 上以 fp32 运行时，移植版与上游引擎逐位
一致：每次前向传播的 logits 和隐藏状态，以及生成的 token、框的浮点数和掩码字节。

## 导出

仅支持推理：`train()`、`val()`、`export()` 和 `chat()` 会抛出异常。

## 许可证

<provenance-box>

Falcon Perception 的代码和权重都采用 Apache-2.0 许可，出自
[技术创新研究院](https://github.com/tiiuae/Falcon-Perception)。
允许商用；TII 的 NOTICE 要求面向公众的作品保留该声明并引用这个项目。AnyUp 特征
上采样器只在 600m 的 segment 任务中使用，出自 AnyUp 作者，采用 CC BY 4.0 许可，
允许在署名的前提下商用。LibreYOLO 不再分发这些权重。

</provenance-box>

## 引用

<citation-block />
