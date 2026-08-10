---
title: InternVL3
families: [internvl3]
seo_title: "InternVL3：在 LibreYOLO 里做开放词汇检测"
description: "在 LibreYOLO 里用 InternVL3 做开放词汇目标检测。用任意文本标签跑预测；不支持训练、验证和导出。"
lead: "InternVL3 是 OpenGVLab 发布的原生多模态大语言模型，在同一个预训练阶段里联合学习视觉和语言。LibreYOLO 把它包装成一个开放词汇目标检测器：任意一组文本标签都能当作类别集合，没有固定的 head，也不需要微调。"
keywords: [InternVL3, InternVL, "internvl3 目标检测", 视觉语言模型, 开放词汇检测, VLM, OpenGVLab, LibreVLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # 开放词汇：任何词都可以，不是一个固定的类别 head。设置之后会一直
        # 沿用到后面每一次 predict()/track() 调用，直到再设置一次
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 原始对话
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # 检测这层便利封装底下的逃生口：自由形式的提问、计数，或者任何
        # 检测框封装覆盖不到的提示词
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
---

## 安装

InternVL3 需要 `vlm` extra，它会装上 chat 模板骨干所用到的 `transformers`。

```bash
pip install "libreyolo[vlm]"
```

## 预测

`LibreInternVL3` 是一个 Python 类，不是一个 `.pt` 检查点（checkpoint）：它不通过
`LibreYOLO()` 工厂加载，`libreyolo` CLI 也解析不到它。`LibreVLM(...)` 工厂
（`from libreyolo import LibreVLM`）也能通过别名找到这个家族，例如
`LibreVLM("internvl3-2b")`；下面用的类就是它构造出来的那个。权重来自 OpenGVLab
自己的 `-hf` Hugging Face 仓库，不是 LibreYOLO 镜像；首次调用会把它们下载并缓存
到本地，并在下载之前为受限（gated）的 Qwen 权重记录一次性的许可证提示。

<code-tabs name="predict" />

`result.boxes` 和其他家族一样承载解析出来的检测结果。置信度是一个占位值：
InternVL3 不输出逐框的分数，所以每个检测拿到的都是同一个常数置信度，`conf=` 只会
丢掉低于这个常数的行，它不会给它们排序。`iou` 会丢掉同一类别里重叠超过给定值的
近重复检测框，那是贪心解码把同一个物体重复输出的副作用；它不是一次按类别的 NMS。
不调用 `set_classes()` 时，词汇表默认就是 COCO-80 的类别名。数据源、流式处理和
结果处理见[预测](/docs/predict)。

## 变体

三种尺寸：1b、2b 和 8b，全部是 OpenGVLab 原生的 `-hf` 检查点（一个 Qwen LLM
骨干，不是最初那篇 InternVL 论文描述的双塔架构）。LibreYOLO 的基准测试工具没有
测过这个家族，所以没有已发布的精度数字可以拿来比较它们；按你自己的算力预算挑一个
尺寸。

LibreYOLO 只把这个家族开放给预测。`train()`、`val()` 和 `export()` 都会抛出
`NotImplementedError`：微调请在上游做，然后把结果加载进来，数据集验证被跳过是
因为占位的置信度会让 COCO mAP 产生误导，而导出对一个没有 state dict 可追踪的
生成式模型来说超出了范围。

## 许可证

<provenance-box>

InternVL3 自己的代码是 MIT，宽松许可，可以用在商业和闭源产品里。这个家族加载的
`-hf` 检查点带着一个 Qwen LLM 骨干，它们单独采用阿里云的 Qwen License：可以免费
使用、修改和再分发，但要求标注「Built with Qwen」或「Improved using Qwen」，并且
商用有每月一亿活跃用户的上限，超过之后需要阿里巴巴另行授权。LibreYOLO 不托管也不
再分发这些权重：`LibreInternVL3` 在首次运行时直接从 Hugging Face 上的
`OpenGVLab/InternVL3-<size>-hf` 下载对应的尺寸，并在那次下载之前为 Qwen License
记录一次性提示。

</provenance-box>

## 引用

<citation-block />
