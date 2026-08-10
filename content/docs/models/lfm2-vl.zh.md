---
title: LFM2-VL
families:
  - lfm2vl
seo_title: LFM2-VL：在 LibreYOLO 里做开放词汇检测
description: 用 LibreYOLO 里的 LFM2-VL 在端侧做开放词汇目标检测。用任意文本标签做预测；不支持训练、验证和导出。
lead: >-
  LFM2-VL 是 Liquid AI 发布的一个紧凑的端侧视觉语言模型。LibreYOLO
  把它包装成开放词汇目标检测器：任何一份文本标签列表都能成为类别集合，不需要固定的 head，也不需要微调。
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - 视觉语言模型
  - 开放词汇目标检测
  - VLM
  - 端侧 vlm
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # 开放词汇：任何词都可以，不是固定的类别 head。它粘住不放，后面每一次
        # predict()/track() 调用都沿用，直到再次设置
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 直接对话
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # 检测这层便利封装底下的逃生通道：自由形式的提问、计数，或者任何检测框
        # 封装覆盖不到的提示词
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## 安装

LFM2-VL 需要 `vlm` extra，它会为 chat 模板骨干装上 `transformers`。

```bash
pip install "libreyolo[vlm]"
```

## 预测

`LibreLFM2VL` 是一个 Python 类，不是 `.pt` 检查点（checkpoint）：它不通过
`LibreYOLO()` 工厂加载，`libreyolo` CLI 也解析不了它。`LibreVLM(...)` 工厂
（`from libreyolo import LibreVLM`）也能通过别名到达这个家族，例如
`LibreVLM("lfm2-vl-450m")`；下面用到的这个类就是它构造出来的东西。权重来自
Liquid AI 自己的 Hugging Face 仓库，而不是 LibreYOLO 的镜像；首次调用会把它们下
载并缓存在本地，并在此之前记录一条一次性的许可提示。

<code-tabs name="predict" />

`result.boxes` 和其他任何家族一样，装着解析出来的检测结果。置信度是个占位值：
LFM2-VL 不输出逐框分数，所以每个检测结果拿到的都是同一个常数置信度，`conf=` 只
会丢掉低于这个常数的行，并不会给它们排序。`iou` 会把同一类别里重叠度超过给定值
的近重复检测框丢掉，这是贪心解码重复输出同一个物体带来的副作用；它不是一趟按类
别做的 NMS。不调用 `set_classes()` 时，词汇表默认就是 COCO-80 的类别名。数据源、
流式处理和结果处理见[预测](/docs/predict)。

## 变体

两个尺寸：450m 和 1.6b，都来自 Liquid AI 的 LFM2.5-VL 发布版，为端侧部署而造。
LibreYOLO 的基准测试工具还没有测过这个家族，所以没有公开的精度数字可以拿来比较
它们；按你自己的算力预算挑一个尺寸。

LibreYOLO 只把这个家族开放给预测。`train()`、`val()` 和 `export()` 全都抛出
`NotImplementedError`：要微调就在上游做，然后把结果加载进来；数据集验证被跳过，
是因为占位的置信度会让 COCO mAP 产生误导；导出则不在范围内，因为这是一个生成式
模型，没有 state dict 可以追踪。

## 许可证

<provenance-box>

LFM Open License v1.0 允许商用、复制和修改，但只在年营收 1000 万美元这个阈值以
下有效；达到或超过这个阈值的法律实体，在商用上根本不受本协议许可，必须直接联系
Liquid AI。符合条件的非营利组织在非商业或研究用途上不受这个阈值限制。LibreYOLO
不分发任何 LiquidAI 源代码，因为模型是通过采用 Apache-2.0 许可的 `transformers`
库加载的，也不托管或转存这些权重：`LibreLFM2VL` 在首次运行时直接从 Liquid AI 自
己的 Hugging Face 仓库下载对应的尺寸，并在那次下载之前记录一条一次性提示。

</provenance-box>

## 引用

<citation-block />
