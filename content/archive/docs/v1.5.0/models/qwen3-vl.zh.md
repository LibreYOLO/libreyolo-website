---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: Qwen3-VL：在 LibreYOLO 里做开放词汇检测
description: 在 LibreYOLO 里用 Qwen3-VL：安装、设定开放词汇，用阿里巴巴这个采用 Apache-2.0 许可的视觉语言模型做预测或对话。
lead: >-
  Qwen3-VL 是阿里巴巴的视觉语言模型，原生支持 2D grounding。LibreYOLO
  把它包装成一个开放词汇目标检测器，并直接暴露它的自由形式对话：给一份类别列表就做检测，或者直接问它一个问题。
keywords:
  - Qwen3-VL
  - Qwen
  - qwen3-vl 目标检测
  - qwen3-vl python
  - 视觉语言模型
  - 开放词汇检测
  - 阿里巴巴
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 对话
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # 检测这层便利封装底下的逃生口：任何问题都可以问，

        # 不只是检测框查询

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## 安装

Qwen3-VL 属于 LibreYOLO 的「VLM 当检测器」这一层，它和那些基于检查点
（checkpoint）的家族是相互独立的两块产品面，有自己的工厂。它需要 `vlm` extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。`LibreVLM()` 不带参数调用时
默认是 Qwen3-VL-4B。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 工厂加载，而不是 `LibreYOLO()`：VLM 家族不声明检查点
加载器，所以其他模型页面上讲的那套按文件后缀路由在这里不适用。`set_classes()`
设定要让 Qwen3-VL 去找的词汇表；它粘住不放，会在之后每一次 `predict()`/`track()`
调用里一直生效，直到你再设一次。每个检测结果带的都是同一个占位置信度，所以按
`conf` 过滤是全有或全无，而不是一种排序；`iou` 对这个家族确实有作用，一旦后出现
的同类别检测框和某个已经保留下来的框重叠超过阈值，它就会被丢掉，因为一个会重复
输出的生成器可能会对同一个物体吐出近乎重复的检测框。和 Florence-2、Kosmos-2 不
同，Qwen3-VL 还能通过 `chat()` 回答自由形式的问题，也就是 `LibreVLM` 工厂那一页
上写的同一个逃生口。LibreYOLO 的 CLI 不覆盖这一层：它没有
`libreyolo predict model=...` 这种形式。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

三种尺寸：Qwen3-VL-2B-Instruct、Qwen3-VL-4B-Instruct 和 Qwen3-VL-8B-Instruct，
分别用 `LibreVLM("qwen3-vl-2b")`、`LibreVLM("qwen3-vl-4b")` 和
`LibreVLM("qwen3-vl-8b")` 加载。三者都声明 1024 px 的标称输入，但实际传给网络的
画布由 Qwen 处理器自己的 smart-resize 决定，所以这个数字并不像本站其他家族那样是
一个固定的工作分辨率。LibreYOLO 还没有发布过比较这三种尺寸精度的基准测试。

LibreYOLO 不训练、不验证、也不导出 Qwen3-VL：`train()`、`val()` 和 `export()`
对这一层里的每个家族都抛出 `NotImplementedError`（见上面的支持层级）。如果你需要
把一份自定义词汇表固化进模型，就在上游微调 Qwen3-VL，再加载得到的权重；用肉眼
检查 `predict()` 的输出，而不是跑一遍 COCO 式的验证，因为每个检测结果带的都是同
一个占位置信度。

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
