---
title: DeepSeek-OCR
families:
  - deepseekocr
seo_title: DeepSeek-OCR 与 DeepSeek-OCR-2 Python 用法：文档 OCR 转 markdown
description: >-
  在 LibreYOLO 里运行 DeepSeek-OCR 和 DeepSeek-OCR-2：输出带检测框的文本行，或把整页转成 markdown。采用
  MIT 和 Apache-2.0 许可，不运行远程代码。
lead: >-
  DeepSeek-OCR 是 DeepSeek 面向文档的 3B 混合专家模型。LibreYOLO 通过内置的 PyTorch 移植版在 ocr 任务上运行
  DeepSeek-OCR 和 DeepSeek-OCR-2：输出带检测框的文本行、版面块，或把整页转成 markdown。
keywords:
  - DeepSeek-OCR
  - DeepSeek-OCR-2
  - 文档 OCR
  - OCR 转 markdown
  - 视觉语言模型 vlm
  - 混合专家 moe
  - deepseek ocr python
  - LibreVLM
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr")      # 权重在首次使用时下载
        result = model.predict("page.png")     # 一张扫描或渲染出来的页面

        for text, quad in zip(result.ocr.texts, result.ocr.polygons):
            print(text, quad.tolist())         # 每个文本行一个区域
    - label: Markdown
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr", prompt="markdown")
        result = model.predict("page.png")

        print(result.ocr.markdown)             # 整页的 markdown
        print(result.ocr.labels)               # 每个块的版面类别
    - label: DeepSeek-OCR-2
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr-2")     # 分辨率为 "gundam" 或 "base"
        result = model.predict("page.png")
        print(result.ocr.texts)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr")
        metrics = model.val(data="ocr_dataset/")   # 检测 hmean、端到端 F1、1-NED
source_hash: 51211ee8cf74b682
---

## 安装

DeepSeek-OCR 属于 LibreYOLO 的 VLM 层级，需要 `vlm` extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从逐字节一致的 LibreYOLO 镜像下载，即
`LibreYOLO/LibreDeepSeekOCR1` 和 `LibreYOLO/LibreDeepSeekOCR2`，固定到某个
revision。架构是发布版推理路径的 PyTorch 移植；不运行任何远程代码。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 或 `LibreDeepSeekOCR` 类加载。`prompt` 设置（在构造时
或通过 `set_prompt()`）决定 `Results.ocr` 里放什么：

| `prompt` | `Results.ocr` |
| --- | --- |
| `"ocr"`（默认） | 每个文本行一个区域，`labels` 全为 `text`，`markdown` 为空 |
| `"markdown"` | 每个版面块一个区域，版面类别在 `labels` 里，整页在 `markdown` 里 |
| `"free"` | 没有区域，页面文本在 `markdown` 里 |
| 其他任意字符串 | 模型输出的任何定位（grounding）结果，外加 `markdown` |

检测框以模型的 0 到 999 网格给出，再缩放回原图，成为轴对齐的 4 点四边形，按从上
到下、再从左到右排序。生成式模型不输出分数，所以 `conf` 和 `det_conf` 恒为 1.0。
`model.chat("page.png", "Parse the figure.")` 对任意提示词返回原始生成文本。解码
采用贪心解码，最多生成 8192 个新 token，并像发布版一样使用
`no_repeat_ngram_size=20`。

## 变体

| 名称 | 模型 | 分辨率 | 许可证 |
| --- | --- | --- | --- |
| `deepseek-ocr` | DeepSeek-OCR（尺寸 `1`） | `tiny`（512）、`small`（640）、`base`（1024）、`large`（1280）、`gundam`（默认） | MIT |
| `deepseek-ocr-2` | DeepSeek-OCR-2（尺寸 `2`） | `base`（1024）、`gundam`（默认） | Apache-2.0 |

`gundam` 把一个 1024 px 的全局视图和局部裁剪结合起来：DeepSeek-OCR 上是 2 到 9 个
640 px 的裁剪，DeepSeek-OCR-2 上最多 6 个 768 px 的裁剪。DeepSeek-OCR-2 保留了
解码器和 SAM 阶段，把 CLIP 换成了 Qwen2 编码器；提示词和输出相同。

CUDA 上像发布版一样在 `torch.autocast` 下运行 bf16 权重（不支持 bf16 的 GPU 上用
fp16）。CPU 和 MPS 以 float32 和 eager 注意力运行，约占 13 GB 内存；MPS 尚未在
硬件上运行过。不应用 EXIF 方向信息。在 RTX 3090 上，移植版在 DeepSeek-OCR 的 17 个
用例和 DeepSeek-OCR-2 的 14 个用例上与发布版逐 token 一致。基准测试数字见发布版
论文，LibreYOLO 没有复现。

## 验证

`val()` 运行共用的 OCR 验证器，报告检测 hmean、端到端 F1 和 1-NED。

<code-tabs name="val" />

## 导出

仅支持推理：`train()` 和 `export()` 会抛出异常。

## 许可证

<provenance-box>

DeepSeek-OCR 采用 MIT 许可，DeepSeek-OCR-2 采用 Apache-2.0 许可，代码和权重都是如此，
均出自 [DeepSeek](https://github.com/deepseek-ai/DeepSeek-OCR)。两者都允许商用。
权重是逐字节一致、不需要申请访问的镜像。

</provenance-box>

## 引用

<citation-block />
