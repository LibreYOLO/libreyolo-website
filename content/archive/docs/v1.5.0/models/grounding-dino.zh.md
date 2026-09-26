---
title: Grounding DINO
families:
  - grounding_dino
seo_title: Grounding DINO：在 LibreYOLO 里做开集检测
description: 用 LibreYOLO 里的 Grounding DINO 检测任何用文本描述的物体。装上 openvocab extra，用自由文本词汇表做预测。
lead: >-
  Grounding DINO 是一个开集（open-set）目标检测器，由 IDEA Research
  开发，它拿图像去和一段自由文本提示词打分，而不是去对一个固定的类别列表。LibreYOLO 把它包装成开放词汇检测器层里一个仅支持预测的家族。
keywords:
  - Grounding DINO
  - 开放词汇目标检测
  - grounding dino 文本提示检测
  - 零样本目标检测
  - 开集检测 python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 文本阈值
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf 按检测框分数过滤，text_threshold 按解码出的短语的 token
        # 分数过滤，两者不设置时都默认为 0.25
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## 安装

Grounding DINO 通过 LibreYOLO 的开放词汇检测器层加载，这一层需要 `openvocab`
extra：

```bash
pip install "libreyolo[openvocab]"
```

这个 extra 会装上 `transformers` 和 `timm`，也就是这一层调用的那两个 Hugging
Face 库。

## 预测

Grounding DINO 不是 LibreYOLO 通过 `LibreYOLO()` 加载的检查点（checkpoint）。它
通过同级的 `LibreOpenVocab` 工厂加载，后者在首次使用时下载一份 Hugging Face
快照，并缓存在 `weights/` 下。

<code-tabs name="predict" />

`set_classes()` 设定一个粘住不放的文本词汇表：再调用一次就能替换整个列表，不调用
则保留默认的 COCO-80 标签。Grounding DINO 从自己的文本输出里解码出自由形式的短
语，并自行把它们映射回那个词汇表，归一化之后的精确匹配优先，整 token 的匹配也会
被接受，而有歧义或匹配不上的短语会被丢弃，而不是靠猜，所以 `school bus` 绝不会
被映射成单独的 `bus` 或 `school`。词汇表长到超出文本编码器的 token 上限时，会被
拆成若干个提示词，作为多次独立的前向传播运行，再合并回一组检测结果，并由
`max_det` 封顶。

`iou` 为了 API 兼容性会被接受，但会发出警告并且没有任何作用，因为这里没有任何东
西跑非极大值抑制。`imgsz` 和 `augment=True` 会被直接拒绝：缩放由 `transformers`
的 processor 负责，而测试时增强不在这一层的范围内。对单张图像调用
`predict()` 返回的是一个 `Results`，不是列表；传入一个目录、一个图像列表，或者
对视频源用 `stream=True`，才会得到多个。这个家族没有 CLI 路径，
`libreyolo predict` 只通过 `LibreYOLO()` 加载 `.pt` 检查点，所以
`LibreOpenVocab` 家族从 Python 里运行。数据源类型和流式处理见[预测](/docs/predict)。

## 变体

两个检查点，`t` 和 `b`。`t` 是这一层未指定尺寸时的默认尺寸。两者都通过
`transformers` 的 `GroundingDinoForObjectDetection` 镜像官方的 IDEA Research
发布版，一次性下载到一份保留了上游文件的 LibreYOLO 托管 Hugging Face 快照里。
这个家族目前还没有发布任何精度或延迟数字。

训练、数据集验证和导出都不在这一层的范围内：`train()`、`val()` 和 `export()` 都
无条件抛出 `NotImplementedError`。这是一个围绕已发布检查点的、仅支持预测的封装。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
