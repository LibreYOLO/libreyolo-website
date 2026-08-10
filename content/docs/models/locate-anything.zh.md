---
title: LocateAnything
families: [locateanything]
seo_title: "LocateAnything：开放词汇检测与点定位"
description: "在 LibreYOLO 里用 LocateAnything 做开放词汇检测和点定位。用任意文本标签跑预测；不支持训练、验证和导出。"
lead: "LocateAnything 是英伟达发布的视觉定位（grounding）模型，它并行解码检测框和点，而不是一次只解码一个坐标 token。LibreYOLO 把它包装成一个开放词汇检测器兼点定位器：任意一组文本标签都能当作类别集合，没有固定的 head，也不需要微调。"
keywords: [LocateAnything, NVIDIA, "locateanything 目标检测", 视觉语言模型, 开放词汇检测, "grounding 视觉定位", "点定位 python", VLM, LibreVLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # 开放词汇：任何词都可以，不是一个固定的类别 head。设置之后会一直
        # 沿用到后面每一次 predict()/track() 调用，直到再设置一次
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 点提示
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point" 返回的是每个匹配到的物体一个点，而不是一个检测框。
        # 已经加载好的模型可以用 model.set_task("point") 切换任务
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: 原始对话
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # 检测这层便利封装底下的逃生口：自由形式的提问、计数，或者任何
        # 检测框封装覆盖不到的提示词
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
---

## 安装

LocateAnything 需要 `vlm` extra，它会装上 `transformers`，以及它在 Hugging Face
上的远程代码在加载时会导入的 `decord`、`lmdb` 和 `peft` 这几个包。

```bash
pip install "libreyolo[vlm]"
```

## 预测

`LibreLocateAnything` 是一个 Python 类，不是一个 `.pt` 检查点（checkpoint）：它不
通过 `LibreYOLO()` 工厂加载，`libreyolo` CLI 也解析不到它。`LibreVLM(...)` 工厂
（`from libreyolo import LibreVLM`）也能通过别名找到这个家族，例如
`LibreVLM("locate-anything")`；下面用的类就是它构造出来的那个。加载它会从
Hugging Face 下载并执行英伟达自己的远程模型代码，所以 LibreYOLO 把下载锁定在一个
固定的 commit 上，而不是会变动的 `main` 分支，并在首次下载之前记录一次性的许可证
提示。

<code-tabs name="predict" />

`result.boxes`（任务 `detect`）和 `result.points`（任务 `point`）和其他家族一样
承载解析出来的输出。置信度是一个占位值：LocateAnything 不输出逐框的分数，所以每个
检测拿到的都是同一个常数置信度，`conf=` 只会丢掉低于这个常数的行，它不会给它们
排序。不调用 `set_classes()` 时，词汇表默认就是 COCO-80 的类别名。数据源、流式
处理和结果处理见[预测](/docs/predict)。

## 变体

已发布的尺寸只有一种，3b。两个任务共用同一份权重：`detect`（默认）返回检测框，而
`task="point"` 返回的是每个匹配到的物体一个点，放在 `result.points` 里；已经加载
好的模型可以用 `model.set_task("point")` 在两者之间切换。LibreYOLO 的基准测试工具
没有测过这个家族，所以没有已发布的精度数字可以拿来比较。

LibreYOLO 只把这个家族开放给预测。`train()`、`val()` 和 `export()` 都会抛出
`NotImplementedError`：微调请在上游做，然后把结果加载进来，数据集验证被跳过是
因为占位的置信度会让 COCO mAP 产生误导，而导出对一个没有 state dict 可追踪的
生成式模型来说超出了范围。

## 许可证

<provenance-box>

NVIDIA License 允许使用、复制和修改，但对英伟达及其关联公司以外的任何人，都把这个
模型和任何衍生物限制在非商业用途、研究或评估上：没有营收门槛，也没有付费例外。
LocateAnything-3B 还组合了另外两个各自带许可的组件：一个采用 Qwen Research License
的 Qwen2.5-3B-Instruct 语言骨干，以及一个采用 MIT 许可的 MoonViT-SO-400M 视觉
编码器。LibreYOLO 不托管、不镜像也不再分发其中的任何一部分：`LibreLocateAnything`
在首次运行时直接从 Hugging Face 上的 `nvidia/LocateAnything-3B` 下载权重和所需的
远程代码，并锁定在一个固定的 commit 上。

</provenance-box>

## 引用

<citation-block />
