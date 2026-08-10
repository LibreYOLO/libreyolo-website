---
title: SenseNova-Vision
families: [sensenovavision]
seo_title: "SenseNova-Vision：在 LibreYOLO 里用一个检查点做 7 个任务"
description: "在 LibreYOLO 里用 SenseNova-Vision 做检测、分割、全景分割、姿态、点、深度和 OCR，全部来自同一个提示驱动的生成式检查点。"
lead: "SenseNova-Vision 是一个统一多模态模型，它把视觉任务当成共享解码器上的提示驱动生成：检测框、点、关键点和 OCR 文字以带标签的文本输出，而深度图、掩码图和全景图以解码器渲染出的图像输出。LibreYOLO 通过 LibreVLM 加载它，用这一个 7B 检查点支持七种任务。"
keywords: [SenseNova-Vision, 商汤, 统一多模态模型, Bagel, "多模态大模型 目标检测", 稠密感知, 指代分割, "全景分割 python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() 在同一个已加载的模型上切换任务
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: 指代分割与全景分割
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="segment")
        # 分割是指代式的：它需要一个目标短语，而不是一个类别列表
        model.set_classes(["the person furthest to the right"])
        result = model.predict("street.jpg")
        mask = result.masks.data[0]

        model.set_task("panoptic")
        # 没有设置自定义词汇表时，全景分割会回退到这个检查点调优时用的
        # COCO 全景类别
        result = model.predict("street.jpg")
        segment_map = result.panoptic.data
        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 点、姿态与 OCR
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # 没有设置词汇表时，姿态会回退到 "person"
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
---

## 安装

SenseNova-Vision 需要自己的 extra，它会为这个检查点所需的大模型分发（dispatch）拉入 `accelerate`，并在非 macOS 平台上拉入用于 4-bit 加载的 `bitsandbytes`。

```bash
pip install "libreyolo[sensenova]"
```

检查点镜像在 LibreYOLO 自己的 Hugging Face 组织下，首次使用时自动下载；它采用 CC-BY-NC-4.0 许可，仅限非商业用途，加载器在每一次自动下载之前都会打印这条提示。见下面的许可证一节。

## 预测

<code-tabs name="predict" />

每一次预测都是在共享的 Bagel-MoT 骨干上做一次扩散解码，所以它是一个能力型模型，而不是一个实时模型：单张图像的延迟会明显高于一个专门打造的检测器或分割器。`dtype="auto"`（默认值）会在显存足够的 GPU 上加载 bf16，其他情况下回退到 4-bit NF4 量化，那需要 `bitsandbytes`；传 `dtype="bf16"` 可以在足够大的 GPU 上强制使用全精度。构造时的 `noise_seed=42` 会给扩散采样器设置随机种子，让稠密输出可复现；传 `noise_seed=None` 可以关掉设种子。

七个任务共用一份已加载的检查点：`set_task()` 在它们之间切换，不需要重新加载。`set_classes()` 设置当前生效的词汇表；检测、点、姿态和全景分割接受一个类别列表，而分割是指代式的，需要的正是要分离出来的那个短语。每个任务都返回标准的 `Results` 对象，只是填充的载荷不同：detect 填 `boxes`，point 填 `points`，pose 填 `boxes` 和 `keypoints`，OCR 填 `ocr`，depth 填 `depth_map`，segment 填 `masks`，panoptic 填 `panoptic`（带 `segments_info`）。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 检查点

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
