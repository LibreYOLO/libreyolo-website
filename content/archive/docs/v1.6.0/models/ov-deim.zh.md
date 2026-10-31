---
title: OV-DEIM
families:
  - ov_deim
seo_title: OV-DEIM：在 LibreYOLO 里做开放词汇检测
description: 用 LibreYOLO 里的 OV-DEIM 做实时的 DETR 式开放词汇检测。装上 openvocab extra，用自由文本词汇表做预测。
lead: >-
  OV-DEIM 是一个 DETR 式的开放词汇目标检测器，它把解码器 query 拿去和随库提供的 MobileCLIP
  文本塔输出的文本嵌入向量做匹配。LibreYOLO 把它原生移植成开放词汇检测器层里一个仅支持预测的家族。
keywords:
  - OV-DEIM
  - DEIMv2
  - 开放词汇目标检测
  - 实时目标检测 python
  - 零样本检测 python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 替换词汇表
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # 第二次调用 set_classes() 会把词汇表整个替换掉，
        # 并通过文本塔重新嵌入一遍；空结果是一种合法的结果，
        # 而不是错误
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## 安装

OV-DEIM 通过 LibreYOLO 的开放词汇检测器层加载，这一层需要 `openvocab` extra：

```bash
pip install "libreyolo[openvocab]"
```

和这一层的其余部分不同，OV-DEIM 是 LibreYOLO 的原生移植，而不是一个
`transformers` 封装，它没有对应的 `transformers` 模型类，但同一个 extra 覆盖了它
在预测时需要的 `huggingface_hub`、`safetensors`、`regex` 和 `ftfy` 这几个包。

## 预测

OV-DEIM 不是 LibreYOLO 通过 `LibreYOLO()` 加载的检查点（checkpoint）。它通过同级
的 `LibreOpenVocab` 工厂加载，后者在首次使用时下载一份 Hugging Face 快照，并缓存
在 `weights/` 下。

<code-tabs name="predict" />

`set_classes()` 设定一个粘住不放的文本词汇表：再调用一次就能替换整个列表，不调用
则保留默认的 COCO-80 标签，而空结果是一种合法的结果，而不是错误。每个解码器
query 都以余弦相似度对着随库提供的 MobileCLIP-B(LT) 文本塔输出的文本嵌入向量打
分，这些嵌入向量按当前设定的词汇表在线计算，并一直缓存到词汇表发生变化为止，所以
任意提示词都能用，不需要任何预先算好的嵌入文件。

OV-DEIM 没有文本 token 阈值：只有 `conf` 会过滤检测结果，传入 `text_threshold` 会
抛错。匹配走的是一对一的 top-K 选择，所以这里没有任何东西跑非极大值抑制，而 `iou`
为了 API 兼容性会被接受，但会发出警告并且没有任何作用。`imgsz` 和 `augment=True`
会被直接拒绝：模型自带一个固定的 letterbox 输入，而测试时增强不在这一层的范围内。
对单张图像调用 `predict()` 返回的是一个 `Results`，不是列表；传入一个目录、一个图
像列表，或者对视频源用 `stream=True`，才会得到多个。这个家族没有 CLI 路径，
`libreyolo predict` 只通过 `LibreYOLO()` 加载 `.pt` 检查点，所以 `LibreOpenVocab`
家族从 Python 里运行。数据源类型和流式处理见[预测](/docs/predict)。

每次调用 `predict()` 也都会跑一遍随库提供的 MobileCLIP-B(LT) 文本塔，把当前的词汇
表嵌入进去；这会给条款添上什么，见许可证一节。

## 变体

三个检查点，`s`、`m` 和 `l`。`s` 是这一层未指定尺寸时的默认尺寸。和这一层的其余
部分不同，OV-DEIM 是原生移植，而不是一个 `transformers` 封装：LibreYOLO 把检测器
模块以和上游代码相同的 Apache-2.0 许可随库提供，并复用了已经为 DEIMv2 家族做好的
DINOv3 骨干适配器。`l` 这个检查点的骨干是一个 DINOv3-S 微调版本，单独采用 Meta 的
DINOv3 License。这个家族目前还没有发布任何精度或延迟数字。

训练、数据集验证和导出都不在这一层的范围内：`train()`、`val()` 和 `export()` 都
无条件抛出 `NotImplementedError`。这是一个围绕已发布检查点的、仅支持预测的封装。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

OV-DEIM 给每一次预测调用都叠上了三份上游许可：检测器权重采用 OV-DEIM 自己的
CC-BY-NC-4.0，在线的文本塔采用 Apple 的 Machine Learning Research Model 许可
（仅限研究用途），而 `l` 这个检查点还带着一个 DINOv3-S 骨干微调版本，采用 Meta
的 DINOv3 License。三份许可证文本都随 LibreYOLO 的权重仓库一起发布。

</provenance-box>

## 引用

<citation-block />
