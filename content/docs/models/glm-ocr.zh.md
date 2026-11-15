---
title: GLM-OCR
families:
  - glmocr
seo_title: GLM-OCR Python 用法：把文档页面解析成 markdown
description: >-
  在 LibreYOLO 里运行 Z.ai 的 GLM-OCR：PP-DocLayoutV3 做版面分析，再由一个 0.9B
  识别模型读取文本、表格（HTML）和公式（LaTeX）。采用 MIT 和 Apache-2.0 许可。
lead: >-
  GLM-OCR 是 Z.ai 的两阶段文档解析器。PP-DocLayoutV3 找出版面区域及其阅读顺序，0.9B 的 GLM-OCR
  识别模型把每个区域读成文本、HTML 表格或 LaTeX 公式。LibreYOLO 为 ocr 任务移植了围绕这两个网络的官方 SDK 流水线。
keywords:
  - GLM-OCR
  - Z.ai
  - zai-org
  - 文档解析
  - OCR 转 markdown
  - 表格识别
  - 公式识别
  - glm ocr python
  - PP-DocLayoutV3
  - LibreVLM
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")            # 权重在首次使用时下载
        result = model.predict("page.png")      # 一张扫描或渲染出来的页面

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])             # 每个区域一个块，按阅读顺序
        print(result.ocr.markdown)              # 整页的 markdown
    - label: 单个元素
      language: python
      code: >
        from libreyolo import LibreVLM


        # 跳过版面检测，直接读取一张预先裁好的表格

        model = LibreVLM("glm-ocr", use_layout_detection=False,
        prompt_label="table")

        result = model.predict("table.png")

        print(result.ocr.texts[0])              # HTML 表格
    - label: 信息抽取
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("glm-ocr")

        reply = model.chat("id_card.png", 'Extract as JSON: {"name": "", "date":
        ""}')

        print(reply)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")
        metrics = model.val(data="ocr_dataset/")   # 检测 hmean、端到端 F1、1-NED
source_hash: 49c7d7cc949d1a45
---

## 安装

GLM-OCR 属于 LibreYOLO 的 VLM 层级，需要 `vlm` extra。两个网络都通过 transformers
的原生类加载，需要 transformers 5.3 或更高版本。

```bash
pip install "libreyolo[vlm]"
```

## 预测

两个网络的权重在首次使用时从同一个固定版本的镜像 `LibreYOLO/LibreGLMOCR0.9b` 下载。
不运行任何远程代码。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 或 `LibreGLMOCR` 类加载。`Results.ocr` 按阅读顺序为每个
版面区域存一个块：`texts`、`polygons`（写成 4 点四边形的轴对齐框）、`labels`（版面
类别，例如 `text`、`table` 或 `formula`），以及 `markdown` 里的整页内容，格式与官方
SDK 一致（标题、`\tag{}` 公式编号、连字符和项目符号修正）。与上游一样，页眉、页脚、
页码、脚注和侧边文字会被丢弃；图片和图表不会被读取。

`Results.ocr.conf` 是生成 token 概率的几何平均值，`det_conf` 是版面分数。`conf=`
不会过滤区域；版面使用上游阈值 0.3，可用 `layout_threshold=` 修改。解码遵循 SDK：
贪心解码，重复惩罚 1.1，最多 8192 个新 token。每个裁剪在识别前会像 SDK 发送时那样
经过一次 JPEG 编解码；`image_format=None` 可以跳过这一步。

`use_layout_detection=False` 配合 `prompt_label=`（`text`、`table` 或 `formula`）
读取预先裁好的元素。这是 LibreYOLO 提供的便利功能，SDK 里没有对应项。`chat()` 覆盖
JSON 信息抽取类的提示词。

## 变体

只有一个尺寸，`0.9b`：一个 CogViT 视觉编码器加一个 GLM-0.5B 解码器。版面阶段以
800 px 运行。CUDA 上识别模型用 bf16（不支持 bf16 时用 fp16），版面检测器用 fp32；
CPU 和 MPS 上两者都用 fp32。PDF 输入、多边形裁剪、图片导出和托管 API 客户端没有移植。

## 验证

`val()` 通过模型自己的 `predict` 运行共用的 OCR 验证器。

<code-tabs name="val" />

## 导出

仅支持推理：`train()` 和 `export()` 会抛出异常。

## 许可证

<provenance-box>

按 [Z.ai](https://github.com/zai-org/GLM-OCR) 模型卡的声明，GLM-OCR 识别模型的
权重采用 MIT 许可。PP-DocLayoutV3 版面检测器采用 Apache-2.0 许可（PaddlePaddle
Authors），LibreYOLO 移植的 SDK 流水线也采用 Apache-2.0 许可。允许商用。权重是
不需要申请访问的镜像。

</provenance-box>

## 引用

<citation-block />
