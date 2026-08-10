---
title: Kosmos-2
families: [kosmos2]
seo_title: "Kosmos-2：LibreYOLO 里的视觉定位目标检测"
description: "在 LibreYOLO 里使用 Kosmos-2：安装、设定开放词汇，用微软这个 MIT 许可的模型预测定位出的检测框。"
lead: "Kosmos-2 是微软的视觉定位（grounding）模型：它先给图像生成一段描述，再用检测框把描述里的每个名词短语定位出来。LibreYOLO 把它包装成一个开放词汇目标检测器：在预测时提供类别列表即可。"
keywords: [Kosmos-2, "视觉语言模型", "grounding 视觉定位", "开放词汇检测", "微软 kosmos-2", VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 视频
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # 库接受的任何输入源：文件、文件夹、URL、摄像头索引、
        # RTSP 流，或者一个 .streams 列表
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
---

## 安装

Kosmos-2 属于 LibreYOLO 的 VLM-as-detector 这一档，是和基于检查点（checkpoint）
的家族分开的另一块产品面，有自己的工厂函数。它需要 `vlm` 这个 extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。LibreYOLO 直接加载微软自己的
`microsoft/kosmos-2-patch14-224` 仓库；和 Florence-2 不同，这里不需要社区的二次
上传。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 工厂函数加载，而不是 `LibreYOLO()`：VLM 家族没有声明
检查点加载器，所以其他模型页上讲的按文件后缀分发在这里不适用。`set_classes()`
设定要求 Kosmos-2 去找的词汇；它是粘性的，会在之后每一次 `predict()`/`track()`
调用中持续生效，直到你再次设定。Kosmos-2 定位的是名词短语，而不是精确匹配一个
标签，所以 LibreYOLO 的包装层接受部分匹配：一个名为 `"boat"` 的类别也能匹配到
「the boats」这样生成出来的短语。每个检测结果带的都是同一个占位置信度，所以 `conf`
过滤是全有或全无，而不是一个排序；`iou` 在这里也没有作用，因为包装层直接用定位到
的实体构建检测列表，没有去重这一步。`chat()` 会抛出 `NotImplementedError`，因为
驱动 Kosmos-2 的是一个 `<grounding>` 提示词，而不是聊天模板。LibreYOLO 的 CLI
不覆盖这一档：没有对应的 `libreyolo predict model=...` 形式。输入源、流式处理和
结果处理见[预测](/docs/predict)。

## 变体

只有一个尺寸：`kosmos-2-patch14-224`，输入 224 px，用 `LibreVLM("kosmos-2")`
加载。它是 2023 年前后的模型，LibreYOLO 自己的包装层也注明，它的定位比这一档里
更新的那些检测器要粗。

LibreYOLO 不训练、不验证、也不导出 Kosmos-2：对这一档里的每个家族，`train()`、
`val()` 和 `export()` 都会抛出 `NotImplementedError`（见上面的支持档位）。如果
你需要把自定义词汇固化进模型，请在上游微调 Kosmos-2，再加载得到的权重；由于每个
检测结果带的都是同一个占位置信度，用眼睛看 `predict()` 的输出，而不是跑一遍 COCO
式的验证。

## 许可

<provenance-box></provenance-box>

## 引用

<citation-block />
