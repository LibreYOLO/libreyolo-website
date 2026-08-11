---
title: 开放词汇检测
seo_title: LibreYOLO 中的开放词汇检测
description: >-
  在 LibreYOLO 中按文本词汇表检测目标。通过 LibreOpenVocab 加载 Grounding DINO、OWLv2、OMDet-Turbo
  或 OV-DEIM，并在运行时设置类别。
lead: >-
  开放词汇检测把检查点（checkpoint）固定的类别列表换成你在调用时自己选的词。在 LibreYOLO 里它不是一个单独的任务：它就是 detect
  任务，只是由一个独立的模型层提供，通过 LibreOpenVocab 工厂函数而不是 LibreYOLO 加载。
keywords:
  - 开放词汇目标检测
  - 零样本目标检测
  - 开放集检测
  - grounding dino python
  - owlv2
  - omdet turbo
  - 文本提示词检测
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
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 切换词汇表
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes 是黏性的：一直生效到下一次调用它
        # 标签在转为小写并去掉冠词后必须唯一
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Grounding DINO 文本阈值
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf 按检测框分数过滤，text_threshold 按解码短语的
        # token 分数过滤，两者不设置时都默认为 0.25，只有 Grounding
        # DINO 接受 text_threshold，其他家族会报错
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## 定义

开放词汇检测返回的是普通的检测 `Results`：检测框、置信度和类别索引，`result.names`
把这些索引映射回你给出的那些字符串。变的是类别列表从哪里来。
传统检测器针对一组固定的类别训练，永远不可能输出这组之外的类别。
这些模型在推理时把词汇表当作文本接收，所以 `set_classes(["forklift", "safety cone"])`
就足以让它们成为类别。

LibreYOLO 没有 `open-vocabulary` 任务键。这些模型和其他检测器一样声明
`SUPPORTED_TASKS = ("detect",)`。区别在加载路径：它们是 Hugging Face 快照，
而不是 LibreYOLO 的 state-dict 检查点，所以它们不走 `LibreYOLO()` 工厂函数，
而是通过 `LibreOpenVocab()` 构造。这个工厂函数是 `LibreSAM()` 和 `LibreVLM()`
的同级，不是 `LibreYOLO()` 的替代。

分数是真正的检测分数，而不是事后解析一段生成出来的图像描述得到的。
每个家族都拿图像区域去和每条提示词的文本嵌入向量比对打分。

## 模型

这一层由四个家族组成，它们全都只支持预测。用别名通过 `LibreOpenVocab` 加载其中任何一个。

[Grounding DINO](/docs/models/grounding-dino)，来自 IDEA Research，有 `t` 和 `b`
两种尺寸。它是这一层的默认模型，也是唯一接受 `text_threshold` 的家族——那是对解码短语的
token 分数的第二道截断。

[OWLv2](/docs/models/owlv2)，来自 Google Research，有 `b16` 和 `l14` 两种尺寸。
它把图像区域和一个 CLIP 式编码器产出的文本嵌入向量比对打分。

[OMDet-Turbo](/docs/models/omdet-turbo)，来自 Om AI Lab，只有一种 `t` 尺寸。
它把类别嵌入向量和语言任务提示词解耦，也是这里唯一在自己的后处理里抑制重叠检测框的家族，
所以 `iou=` 会生效。

[OV-DEIM](/docs/models/ov-deim)，有 `s`、`m` 和 `l` 三种尺寸，是一个 DETR 式检测器，
把解码器查询与随包附带的 MobileCLIP 文本塔产出的文本嵌入向量做匹配。它是一对一匹配加
top-K 选择，所以任何地方都不跑 NMS。

OV-DEIM 的权重是这一层里受限的那一个。检测器权重采用 CC-BY-NC-4.0 许可，不可商用。
随包附带的文本塔采用 Apple 的 Machine Learning Research Model 许可，仅限研究使用。
`l` 检查点还多了一个在 Meta 的 DINOv3 License 之下的 DINOv3-S 骨干微调。
三份许可证全文都随权重仓库一起提供，库在解析权重时、也就是模型构建之前，会记录同样的摘要。
部署它之前先读 [OV-DEIM](/docs/models/ov-deim)。

这一层需要一个额外依赖：

```bash
pip install "libreyolo[openvocab]"
```

它涵盖了三个封装家族所需的 `transformers` 和 `timm`，以及 OV-DEIM 作为原生移植所需的
`huggingface_hub`、`safetensors`、`regex` 和 `ftfy` 这几个包。

还有第二层也接受文本词汇表：`LibreVLM()` 加载生成式视觉语言模型，比如
[Qwen3-VL](/docs/models/qwen3-vl) 和 [Florence-2](/docs/models/florence-2)，
并把它们的输出转成同样的 `Results`。它共用 `set_classes()` 这套接口。
区别在于检测框由什么产生：本页上的这些家族是直接给出分数的判别式检测器，
而 VLM 那一层是生成出来的。

## 预测

<code-tabs name="predict" />

`set_classes()` 接收一个非空的标签字符串列表，并一直生效到再次调用它。
标签在转为小写并去掉开头的冠词后必须唯一，所以 `"a bus"` 和 `"bus"`
不能共存于同一个词汇表。多词短语和别的标签没有区别，每个家族在分词前都会把这个列表
转成自己的文本输入，所以 `"traffic cone"` 和 `"cone"` 是不同的查询。

有三个预测参数在这里的行为和原生检测器不同。`imgsz=` 会被拒绝，
因为这些家族的缩放由 processor 负责。`augment=True` 会被拒绝，
因为测试时增强（TTA）不在这一层的范围内。`iou=`
只对那个在自己的 processor 里做抑制的家族生效；在什么都不抑制的地方，传了它会告警并被忽略。

不设置时，`conf` 取所加载家族自己的默认值，而不是 `predict()` 通常的 0.25，
而且这个默认值在这一层里并不统一。在同一张图像上对比两个家族时，显式设置它。

整层的 `track()` 都会报错。改成逐帧调用 `predict()`。关于输入源、流式处理和结果处理，
见[预测](/docs/predict)。

## 训练

这一层里没有哪个家族能在 LibreYOLO 内部训练。`train()` 会报错：在上游微调，
再加载得到的权重。传给 `set_classes()` 的词汇表是唯一能改变已加载模型检测内容的设置。

## 验证

这一层没有验证器，`val()` 会报错。开放词汇验证需要一个专用的验证器，
因为标准的检测验证器把图像张量直接喂给模型，而这些家族需要的是连同图像一起构建的文本条件输入。

## 导出

导出不在这一层的范围内，`export()` 会报错。这些模型在 PyTorch 里通过 `predict()` 运行。
