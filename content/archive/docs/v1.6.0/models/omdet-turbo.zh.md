---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: OMDet-Turbo：在 LibreYOLO 里做实时零样本检测
description: 用 LibreYOLO 里的 OMDet-Turbo 做实时开放词汇检测。装上 openvocab extra，用自由文本词汇表做预测。
lead: >-
  OMDet-Turbo 是一个实时开放词汇目标检测器，由 Om AI Lab 开发，它把类别嵌入向量和语言任务提示词解耦。LibreYOLO
  把它包装成开放词汇检测器层里一个仅支持预测的家族。
keywords:
  - OMDet-Turbo
  - OmDet
  - 开放词汇目标检测
  - 实时目标检测 python
  - 零样本目标检测
  - 文本描述检测物体
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 自定义 NMS 阈值
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo 是这一层里唯一真正采纳 iou= 的家族：它自己的
        # 后处理把抑制阈值当成一个参数，iou= 不设置时默认为 0.5
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## 安装

OMDet-Turbo 通过 LibreYOLO 的开放词汇检测器层加载，这一层需要 `openvocab`
extra：

```bash
pip install "libreyolo[openvocab]"
```

这个 extra 会装上 `transformers` 和 `timm`，也就是这一层调用的那两个 Hugging
Face 库；OMDet-Turbo 的 Swin 骨干通过 `transformers` 的 `TimmBackbone` 封装加载。

## 预测

OMDet-Turbo 不是 LibreYOLO 通过 `LibreYOLO()` 加载的检查点（checkpoint）。它通过
同级的 `LibreOpenVocab` 工厂加载，后者在首次使用时下载一份 Hugging Face 快照，并
缓存在 `weights/` 下。

<code-tabs name="predict" />

`set_classes()` 设定一个粘住不放的文本词汇表：再调用一次就能整个替换这个列表，不
调用则保留默认的 COCO-80 标签，而空结果是一个有效的结果，不是错误。和 Grounding
DINO 不同，OMDet-Turbo 把类别嵌入向量和语言任务提示词解耦，所以 `transformers`
的后处理返回的标签能直接映射回查询用的那个类别列表，不需要短语消歧这一步。

OMDet-Turbo 没有文本 token 阈值：只有 `conf` 过滤检测结果，传入 `text_threshold`
会抛错。它是这一层里唯一在 `post_process_grounded_object_detection` 内部跑自己的
非极大值抑制的家族，所以 `iou` 在这里会被真正采纳，而不是被警告忽略。`imgsz` 和
`augment=True` 会被直接拒绝：缩放由 `transformers` 的 processor 负责，而测试时增
强不在这一层的范围内。对单张图像调用 `predict()` 返回的是一个 `Results`，不是列
表；传入一个目录、一个图像列表，或者对视频源用 `stream=True`，才会得到多个。这个
家族没有 CLI 路径，`libreyolo predict` 只通过 `LibreYOLO()` 加载 `.pt` 检查点，所
以 `LibreOpenVocab` 家族从 Python 里运行。数据源类型和流式处理见[预测](/docs/predict)。

## 变体

一个检查点，`t`，这一层唯一的尺寸。它通过 `transformers` 的
`OmDetTurboForObjectDetection`，在一个固定住的上游 revision 上镜像
`omlab/omdet-turbo-swin-tiny-hf`；镜像的权重文件和那份上游快照逐字节一致。这个家
族目前还没有发布任何精度或延迟数字。

训练、数据集验证和导出都不在这一层的范围内：`train()`、`val()` 和 `export()` 都无
条件抛出 `NotImplementedError`。这是一个围绕已发布检查点的、仅支持预测的封装。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
