---
title: SmolVLM2
families:
  - smolvlm2
seo_title: SmolVLM2：在 LibreYOLO 里做开放词汇检测
description: >-
  在 LibreYOLO 里用 SmolVLM2：安装、设定开放词汇，用 Hugging Face 这个采用 Apache-2.0
  许可的视觉语言模型做预测或对话。
lead: >-
  SmolVLM2 是 Hugging Face 的小型视觉语言模型。LibreYOLO
  把它包装成一个开放词汇目标检测器，并直接暴露它的自由形式对话：给一份类别列表就做检测，或者直接问它一个问题。
keywords:
  - SmolVLM2
  - smolvlm2 目标检测
  - smolvlm2 python
  - 视觉语言模型
  - 开放词汇检测
  - 小型多模态模型
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 对话
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # 检测这层便利封装底下的逃生口：任何问题都可以问，
        # 不只是检测框查询
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## 安装

SmolVLM2 属于 LibreYOLO 的「VLM 当检测器」这一层，它和那些基于检查点
（checkpoint）的家族是相互独立的两块产品面，有自己的工厂。它需要 `vlm` extra，
这个 extra 还会一并装上 `num2words`，也就是 SmolVLM2 自己的处理器的一个依赖。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 工厂加载，而不是 `LibreYOLO()`：VLM 家族不声明检查点
加载器，所以其他模型页面上讲的那套按文件后缀路由在这里不适用。`set_classes()`
设定要让 SmolVLM2 去找的词汇表；它粘住不放，会在之后每一次 `predict()`/`track()`
调用里一直生效，直到你再设一次。SmolVLM2 在 LibreYOLO 里不需要覆盖解析器：它和
这一层共用的默认形式一样，输出的是聊天模板加 JSON，所以它的检测提示词和检测框
格式并不是家族特有的。每个检测结果带的都是同一个占位置信度，所以按 `conf` 过滤
是全有或全无，而不是一种排序；`iou` 确实有作用，一旦后出现的同类别检测框和某个
已经保留下来的框重叠超过阈值，它就会被丢掉，因为一个会重复输出的生成器可能会对
同一个物体吐出近乎重复的检测框。SmolVLM2 还能通过 `chat()` 回答自由形式的问题，
也就是 `LibreVLM` 工厂那一页上写的同一个逃生口。LibreYOLO 的 CLI 不覆盖这一层：
它没有 `libreyolo predict model=...` 这种形式。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

注册表（registry）里只有一种尺寸：SmolVLM2-500M-Video-Instruct，用
`LibreVLM("smolvlm2-500m")` 加载。作为检测器，SmolVLM2 比这一层里那些专门为
grounding 打造的模型要弱；LibreYOLO 自己的封装把它描述成一个演示，说明一个新家族
不需要任何特例解析就能在这里跑起来，而不是把它当作 LibreYOLO 最强的开放词汇选项。

LibreYOLO 不训练、不验证、也不导出 SmolVLM2：`train()`、`val()` 和 `export()`
对这一层里的每个家族都抛出 `NotImplementedError`（见上面的支持层级）。如果你需要
把一份自定义词汇表固化进模型，就在上游微调 SmolVLM2，再加载得到的权重；用肉眼
检查 `predict()` 的输出，而不是跑一遍 COCO 式的验证，因为每个检测结果带的都是同
一个占位置信度。

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
