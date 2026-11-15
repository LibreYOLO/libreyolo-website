---
title: Falcon OCR
families:
  - falconocr
seo_title: Falcon OCR Python 用法：0.27B 文档 OCR 模型，采用 Apache-2.0 许可
description: >-
  在 LibreYOLO 里运行 TII 的 Falcon OCR：一次把整页转成文本，或配合 PP-DocLayoutV3 逐区域识别。采用
  Apache-2.0 许可，可在 CPU 上运行。
lead: >-
  Falcon OCR 是技术创新研究院（TII）推出的 0.27B 早期融合文档 OCR 模型。一个 transformer
  同时读取图像块和文本，把页面写成文本，公式用 LaTeX，表格用 HTML。LibreYOLO 在 ocr 任务上运行 Falcon OCR 1.5 的
  PyTorch 移植版。
keywords:
  - Falcon OCR
  - TII
  - 文档 OCR
  - 早期融合
  - 小型 OCR 模型
  - OCR 转 markdown
  - falcon ocr python
  - LibreVLM
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr")         # 权重在首次使用时下载
        result = model.predict("page.png")      # 一张扫描或渲染出来的页面

        print(result.ocr.markdown)              # 页面文本
    - label: 带版面分析
      language: python
      code: |
        from libreyolo import LibreVLM

        # PP-DocLayoutV3 按阅读顺序找出区域；每个裁剪
        # 用对应类别的提示词识别
        model = LibreVLM("falcon-ocr", layout=True)
        result = model.predict("page.png")

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])
    - label: 单个元素
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr", prompt="table")
        result = model.predict("table.png")     # 一张裁剪出来的表格
        print(result.ocr.markdown)              # HTML 表格
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr")
        metrics = model.val(data="ocr_dataset/")
source_hash: 3218099fdef6cf7e
---

## 安装

Falcon OCR 属于 LibreYOLO 的 VLM 层级，需要 `vlm` extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 `LibreYOLO/LibreFalconOCR270m` 下载，这是 `tiiuae/Falcon-OCR`
的逐字节一致镜像，固定到某个 revision。发布版通过远程代码运行在 FlexAttention 和
一个 Triton 内核上；LibreYOLO 运行的是不依赖这两者的纯 PyTorch 移植版。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 或 `LibreFalconOCR` 类加载。与上游一样有两种模式：

- 默认：一次读完整页。`Results.ocr` 只有一个覆盖整张图像的区域，标签为 `plain`，
  `Results.ocr.markdown` 是同样的文本。
- `layout=True`：LibreYOLO 的 [PP-DocLayoutV3](/docs/models/pp-doclayoutv3) 移植版
  以 `layout_threshold=0.3` 按阅读顺序找出区域，丢弃嵌套的框，每个裁剪用对应类别
  的提示词识别。每个裁剪一个区域，`labels` 存版面类别，`det_conf` 存版面分数，
  `markdown` 用空行把各段文本连接起来。图片以及上游没有提示词类别的类会被跳过。

`prompt=` 接受一个类别（`plain`、`text`、`table`、`formula`、`caption`、
`footnote`、`list-item`、`page-footer`、`page-header`、`section-header`、
`title`）或一条自定义指令。页面会缩放到两边都在 64 到 1536 px 之间。解码采用贪心
解码，新 token 预算为 8192。生成式模型不输出分数，所以 `conf` 恒为 1.0。

## 变体

只有一个尺寸，`270m`。CUDA 上在 `torch.autocast` 下运行 bf16 权重；CPU 和 MPS 以
float32 运行，权重约 1.1 GB。批大小为 1。在 CPU 上以 float32 运行时，移植版在六个
OCR 测试样例和一张合成的报告页面上与发布版逐 token 一致；CUDA bf16 和版面模式没有
与上游对比过。

## 验证

`val()` 运行共用的 OCR 验证器。

<code-tabs name="val" />

## 导出

仅支持推理：`train()` 和 `export()` 会抛出异常。

## 许可证

<provenance-box>

Falcon OCR 的代码和权重都采用 Apache-2.0 许可，出自
[技术创新研究院](https://github.com/tiiuae/Falcon-Perception)。
允许商用。权重是不需要申请访问、逐字节一致的镜像。可选的版面阶段使用 LibreYOLO
采用 Apache-2.0 许可的 PP-DocLayoutV3 移植版。

</provenance-box>

## 引用

<citation-block />
