---
title: 视觉语言 API
seo_title: LibreVLM API：别名、set_classes 与 chat
description: LibreVLM 工厂、每一个模型别名、set_classes 那份粘性词汇表、set_task、chat 这个逃生口，以及为什么置信度只是一个占位值。
lead: >-
  LibreVLM 加载一个生成式视觉语言模型，并把它当作目标检测器来驱动。类别列表是一个提示词，而不是一个固定的 head，模型返回的 Results
  和其他任何家族返回的一样。
keywords:
  - LibreVLM
  - 视觉语言模型 目标检测
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  别名读自 libreyolo/models/vlm/__init__.py；仓库、尺寸和任务列表读自 libreyolo/models/vlm/
  下的各家族模块，以及 libreyolo/models/sensenova/model.py；调用规则和抛出的异常读自
  libreyolo/models/vlm/base.py，均为 v1.5.0。
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: 检测一份开放词汇
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: 提一个自由形式的问题
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## 安装

这一层需要 `vlm` extra。

<code-tabs name="install" />

## 工厂

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` 是一个别名，不是路径。`**kwargs` 会传到家族的构造函数，它接受 `device`、
`names`（初始词汇表，等价于加载之后调用 `set_classes`）、`prompt`（覆盖检测用的
提示词）和 `max_new_tokens`。未知的别名会抛出 `ValueError`，并列出每一个别名。

<code-tabs name="usage" />

## 别名

| 家族 | 别名 | 尺寸 | 权重 |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | 固定的上游快照 |

默认别名是 `qwen3-vl-4b`。每个家族默认别名对应的尺寸，就是列表里排在最前面的那
个：`qwen3-vl` 解析为 `4b`，`lfm2-vl` 为 `450m`，`internvl3` 为 `2b`，`smolvlm2`
为 `2.2b`，`florence-2` 为 `base`。

`LibreVLM`、`LibreLFM2VL`、`LibreQwen3VL`、`LibreSmolVLM2`、`LibreInternVL3`、
`LibreFlorence2`、`LibreKosmos2`、`LibreLocateAnything` 和 `LibreMODUS`
（也可以写成 `LibreModus`）在包级别导出。

## 任务

大多数家族只服务 `detect`。有两个提供得更多：

| 家族 | 支持的任务 |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

因为任务是提示词驱动的，而不是固化进检查点（checkpoint）里的，所以可以在一个已经
加载好的模型上切换：

```python
model.set_task(task: str) -> LibreVLMModel
```

任务会按家族支持的列表做校验，并且在之后的 `predict()` 和 `track()` 调用中一直
沿用；模型本身会被返回，所以调用可以链起来。

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

设定开放词汇。任何词都可以，因为模型是被这些词提示的，而不是被约束到一个固定的
head 上。列表必须非空，并且忽略大小写比较时各项必须唯一。传入一个裸字符串会抛出
`TypeError`，因为它会被枚举成一堆单字符类别。词汇表是粘性的：加载之后设置一次，
它会一直保留到下一次设置。

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

原始的多模态生成：输入图像和提示词，输出解码后的文本，原样返回。这是检测这层便利
封装底下的逃生口，用于自由形式的提问、计数，或者检测封装覆盖不到的输出格式。
`max_new_tokens` 会回退到家族的 `MAX_NEW_TOKENS`，在基类上是 1024。解码是贪心的，
带一点轻微的重复惩罚。

## 置信度

生成式输出没有经过校准的逐框置信度。这个版本给出一个常数占位值，好让 `predict`、
绘制和 `track` 正常工作，这也让 `conf=` 过滤和 mAP 变得含糊，而不是真有意义。这
也是 `val()` 抛出异常的原因：在占位分数上算 COCO mAP 会产生误导。

## 预测与跟踪

标准的预测接口在这里同样适用，`track()` 也能用，所以一个 VLM 检测器可以像其他任何
家族一样接进同一条流水线。有两条类级别的策略和卷积检测器不同：测试时增强被禁用，
因为多尺度增强对一个固定分辨率的生成器毫无意义；批量预测被关闭，因为生成是自回归
的，而且预处理返回的是一份文本加图像的编码，而不是一个可以堆叠的图像张量。

## 不支持

`train()`、`val()` 和 `export()` 都会抛出 `NotImplementedError`。微调请在上游做，
然后把得到的权重加载进来。

## 远程代码

随库发布的每一个家族都通过原生模型类加载，所以 LibreYOLO 默认不会执行第三方仓库的
代码。确实需要它的家族必须显式选择开启，并固定一个快照修订版；LocateAnything 就是
这样的一个，固定在 commit `c32291ca5e996f5a7a485845b4f57a233936bba0` 上。

LibreMODUS 是检查点结构的一个显式例外：它的别名解析到的是一个装着固定上游文件的
目录，而不是一个 LibreYOLO `.pt`，并且 LibreYOLO 既不给它加 v1.0 元数据，也不再
分发它。
