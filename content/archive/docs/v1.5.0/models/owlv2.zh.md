---
title: OWLv2
families:
  - owlv2
seo_title: OWLv2：在 LibreYOLO 里做零样本目标检测
description: 用 LibreYOLO 里的 OWLv2 检测任何用文本描述的物体。装上 openvocab extra，用自由文本词汇表做预测。
lead: >-
  OWLv2 是一个开放词汇目标检测器，由 Google Research 开发，它把图像区域拿去和一个 CLIP
  式编码器输出的文本嵌入向量打分。LibreYOLO 把它包装成开放词汇检测器层里一个仅支持预测的家族。
keywords:
  - OWLv2
  - OWL-ViT
  - 开放词汇目标检测
  - owlv2 零样本检测
  - 文本提示目标检测 python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 默认词汇表
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # 不调用 set_classes() 就保留这一层默认的 COCO-80 词汇表
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## 安装

OWLv2 通过 LibreYOLO 的开放词汇检测器层加载，这一层需要 `openvocab` extra：

```bash
pip install "libreyolo[openvocab]"
```

这个 extra 会装上 `transformers` 和 `timm`，也就是这一层调用的那两个 Hugging
Face 库。

## 预测

OWLv2 不是 LibreYOLO 通过 `LibreYOLO()` 加载的检查点（checkpoint）。它通过同级的
`LibreOpenVocab` 工厂加载，后者在首次使用时下载一份 Hugging Face 快照，并缓存在
`weights/` 下。

<code-tabs name="predict" />

`set_classes()` 设定一个粘住不放的文本词汇表：再调用一次就能替换整个列表，不调用
则保留默认的 COCO-80 标签。每个标签在送进文本塔之前都会被套进一个固定的提示词模
板，和 `transformers` 的 `Owlv2ForObjectDetection` 训练时的做法保持一致。

OWLv2 没有文本 token 阈值：只有 `conf` 会过滤检测结果，传入 `text_threshold` 会
抛错。`iou` 为了 API 兼容性会被接受，但会发出警告并且没有任何作用，因为这里没有
任何东西跑非极大值抑制。`imgsz` 和 `augment=True` 会被直接拒绝：缩放由
`transformers` 的 processor 负责，而测试时增强不在这一层的范围内。对单张图像调用
`predict()` 返回的是一个 `Results`，不是列表；传入一个目录、一个图像列表，或者对
视频源用 `stream=True`，才会得到多个。这个家族没有 CLI 路径，`libreyolo predict`
只通过 `LibreYOLO()` 加载 `.pt` 检查点，所以 `LibreOpenVocab` 家族从 Python 里运
行。数据源类型和流式处理见[预测](/docs/predict)。

## 变体

两个检查点，`b16`（base，patch 尺寸 16）和 `l14`（large，patch 尺寸 14）。`b16`
是这一层未指定尺寸时的默认尺寸。两者都通过 `transformers` 的
`Owlv2ForObjectDetection` 镜像官方的 Google Research 发布版，一次性下载到一份保
留了上游文件的 LibreYOLO 托管 Hugging Face 快照里。这个家族目前还没有发布任何精
度或延迟数字。

训练、数据集验证和导出都不在这一层的范围内：`train()`、`val()` 和 `export()` 都
无条件抛出 `NotImplementedError`。这是一个围绕已发布检查点的、仅支持预测的封装。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
