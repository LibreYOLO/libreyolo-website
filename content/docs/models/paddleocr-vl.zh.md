---
title: PaddleOCR-VL
families:
  - paddleocrvl
seo_title: PaddleOCR-VL Python 用法：用 0.9B VLM 做文档解析
description: >-
  在 LibreYOLO 里运行 PaddleOCR-VL：PP-DocLayoutV2 做版面分析，再由一个 0.9B
  识别模型读取文本、表格（HTML）、公式（LaTeX）和图表。采用 Apache-2.0 许可。
lead: >-
  PaddleOCR-VL 是飞桨（PaddlePaddle）的两阶段文档解析器。PP-DocLayoutV2 找出 25
  类版面区域及其阅读顺序，PaddleOCR-VL-0.9B 把每个区域读成文本、表格、公式或图表。LibreYOLO 为 ocr 任务移植了围绕这两个网络的
  PaddleX 流水线。
keywords:
  - PaddleOCR-VL
  - PaddleOCR
  - PaddlePaddle
  - 飞桨 OCR
  - 文档解析
  - OCR 转 markdown
  - 表格识别
  - 公式识别
  - ERNIE
  - LibreVLM
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("paddleocr-vl")       # 权重在首次使用时下载
        result = model.predict("page.png")      # 一张扫描或渲染出来的页面

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])             # 每个块一个区域，按阅读顺序
        print(result.ocr.markdown)              # 整页的 markdown
    - label: 单个元素
      language: python
      code: >
        from libreyolo import LibreVLM


        # 跳过版面检测，直接读取一张预先裁好的公式

        model = LibreVLM("paddleocr-vl", use_layout_detection=False,
        prompt_label="formula")

        result = model.predict("formula.png")

        print(result.ocr.texts[0])              # LaTeX
    - label: 图表
      language: python
      code: |
        from libreyolo import LibreVLM

        # 与上游一样，图表默认跳过；显式开启后会把图表读成表格
        model = LibreVLM("paddleocr-vl", use_chart_recognition=True)
        result = model.predict("report.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("paddleocr-vl")
        metrics = model.val(data="ocr_dataset/")   # 检测 hmean、端到端 F1、1-NED
source_hash: 287e66812b8d3b4a
---

## 安装

PaddleOCR-VL 属于 LibreYOLO 的 VLM 层级，需要 `vlm` extra。两个网络都通过
transformers 的原生类加载，需要 transformers 5.3 或更高版本。不需要 PaddlePaddle
本身。

```bash
pip install "libreyolo[vlm]"
```

## 预测

两个网络的权重在首次使用时从同一个固定版本的镜像 `LibreYOLO/LibrePaddleOCRVL0.9b`
下载。不运行任何远程代码。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 或 `LibrePaddleOCRVL` 类加载。`Results.ocr` 按阅读顺序为
每个识别出的块（一个段落、一个标题、一张表格）存一个区域，而不是每个文本行一个：
`texts`、`polygons`（写成 4 点四边形的轴对齐框）、`labels`（版面类别），以及
`markdown` 里的整页内容。表格以 HTML 返回，公式以 LaTeX 返回。图片，以及默认情况下
的图表和印章，不会出现在 `Results.ocr` 和 markdown 里。

`Results.ocr.conf` 是生成 token 概率的几何平均值，`det_conf` 是版面分数。`conf=`
不会过滤区域；版面保留上游的逐类别阈值，可用 `layout_threshold=` 覆盖。识别采用贪心
解码，最多 4096 个新 token，每个区域缩放到 `min_pixels` / `max_pixels`（默认
112896 / 1003520）的范围内。

`use_layout_detection=False` 配合 `prompt_label=`（`ocr`、`table`、`formula` 或
`chart`）读取预先裁好的元素。

## 变体

只有一个尺寸，`0.9b`：一个 NaViT 风格的视觉编码器加 ERNIE-4.5-0.3B 解码器，前面是
PP-DocLayoutV2。CUDA 上识别模型用 bf16（不支持 bf16 时用 fp16），版面检测器用 fp32；
CPU 和 MPS 上两者都用 fp32。在 MPS 上，如果报告缺少某个算子，请设置
`PYTORCH_ENABLE_MPS_FALLBACK=1`。

没有移植的部分：相邻文本块合并、嵌在表格里的图、印章识别和文字检测识别（spotting）、
多边形版面形状、文档矫正和图像导出。markdown 是纯文本变体，没有 HTML 居中。

## 验证

`val()` 运行共用的 OCR 验证器。

<code-tabs name="val" />

## 导出

仅支持推理：`train()` 和 `export()` 会抛出异常。

## 许可证

<provenance-box>

PaddleOCR-VL-0.9B 和 PP-DocLayoutV2 的代码和权重都采用 Apache-2.0 许可，出自
[PaddlePaddle Authors](https://github.com/PaddlePaddle/PaddleOCR)。允许商用。
LibreYOLO 从 PaddleX 移植的流水线同样采用 Apache-2.0 许可。权重是未经修改、
不需要申请访问的镜像。

</provenance-box>

## 引用

<citation-block />
