---
title: Florence-2
families: [florence2]
seo_title: "Florence-2：在 LibreYOLO 里做开放词汇检测"
description: "在 LibreYOLO 里用 Florence-2：安装、设定开放词汇，用微软这个采用 MIT 许可的视觉模型预测检测框。"
lead: "Florence-2 是微软的视觉基础模型，靠一个任务 token 来提示，而不是走一个固定的检测 head。LibreYOLO 把它包装成一个开放词汇目标检测器：类别列表在预测时给出。"
keywords: [Florence-2, 视觉语言模型, 开放词汇检测, "florence 2 python", "florence-2 目标检测", "不用训练的目标检测", grounding, 微软, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 视频
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # 库接受的任何数据源：文件、文件夹、URL、摄像头索引、
        # RTSP 流，或者一个 .streams 列表
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
---

## 安装

Florence-2 属于 LibreYOLO 的「VLM 当检测器」这一层，它和那些基于检查点
（checkpoint）的家族是相互独立的两块产品面，有自己的工厂。它需要 `vlm` extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。LibreYOLO 下载的是
florence-community 对这个检查点的重新上传版本，而不是原始的
`microsoft/Florence-2-*` 仓库；原因见许可证一节。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 工厂加载，而不是 `LibreYOLO()`：VLM 家族不声明检查点
加载器，所以其他模型页面上讲的那套按文件后缀路由在这里不适用。`set_classes()`
设定要让 Florence-2 在图像里找的词汇表；它粘住不放，会在之后每一次
`predict()`/`track()` 调用里一直生效，直到你再设一次。返回的 `Results` 带的
`boxes` 和其他任何家族形状相同，但每个检测结果带的都是同一个占位置信度，所以按
`conf` 过滤是全有或全无，而不是一种排序，`iou` 也没有任何作用：Florence-2 的封装
直接用解析后的任务 token 输出构建检测列表，中间没有去重这一步。这里 `chat()` 会
抛出 `NotImplementedError`，因为 Florence-2 是由 `<OPEN_VOCABULARY_DETECTION>`
这个任务 token 驱动的，而不是靠一套 chat 模板。LibreYOLO 的 CLI 不覆盖这一层：
它没有 `libreyolo predict model=...` 这种形式。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

两种尺寸：Florence-2-base 和 Florence-2-large，都是 768 px，分别用
`LibreVLM("florence-2-base")` 或 `LibreVLM("florence-2-large")` 加载。LibreYOLO
还没有发布过比较两者精度的基准测试。

LibreYOLO 不训练、不验证、也不导出 Florence-2：`train()`、`val()` 和 `export()`
对这一层里的每个家族都抛出 `NotImplementedError`（见上面的支持层级）。如果你需要
把一份自定义词汇表固化进模型，就在上游微调 Florence-2，再加载得到的权重；用肉眼
检查 `predict()` 的输出，而不是跑一遍 COCO 式的验证，因为每个检测结果带的都是同
一个占位置信度。

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
