---
title: PP-OCRv5
families: [ppocr]
seo_title: "PP-OCRv5：在 LibreYOLO 里做文字检测与识别"
description: "用 LibreYOLO 里的 PP-OCRv5 做多语言场景文字 OCR。安装、预测并验证 t 和 l 两个检查点，采用 Apache-2.0 许可。"
lead: "PP-OCRv5 是 PaddleOCR 的文字检测与识别流水线：一个可微二值化（differentiable binarization）检测器定位文字四边形，一个 SVTR/CTC 识别器把它们读出来。LibreYOLO 把它移植到 PyTorch，分成两档。"
keywords: [PP-OCRv5, PaddleOCR, OCR, 文字检测, "文字识别 python", "ocr 中文识别"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: 四边形
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # (N, 4, 2) 多边形，按阅读顺序排列：左上、右上、
        # 右下、左下。检测四边形是真正的多边形（文字带旋转），
        # 所以它们填充的是 result.ocr，而不是 result.boxes
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # 主指标
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
---

## 安装

PP-OCRv5 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

每个检查点（checkpoint）都把检测和识别这两个阶段打包在同一个 `.pt` 文件里，识别字符集和流水线默认值写在检查点元数据里。识别器用一部字典就能读简体中文、繁体中文、英文、日文和拼音。`result.ocr` 是一个 `OCRRegions` 载荷：`.data` 存四点多边形，`.texts` 存转写文本，`.conf` 存每个区域的识别分数，`.det_conf` 存检测分数。多图数据源是顺序跑的：这条两阶段流水线不会跨图像组批。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

两档：`t` 建在更轻的 PP-LCNetV3/PP-OCRv5_mobile 骨干上，面向 CPU 使用；`l` 建在 PP-HGNetV2 服务器端骨干上，精度更高。两档都在固定的长边上限下做检测，并成批识别裁剪图；`rec_batch` 控制每次前向传播送进识别器的裁剪图数量。

## 验证

`val()` 拿一个图像目录加一个 `labels/<split>.jsonl` 文件、或者等价的数据集 YAML 来测量整条流水线，每条标注列出这张图像的文字区域多边形和它们的转写文本。它报告检测 hmean（按 IoU 匹配的查准率/查全率/F1）、端到端 F1（hmean 之上再要求归一化后的转写完全一致，也是这个检查点的 fitness 指标），以及 1-NED，即所有匹配对上的平均归一化编辑距离。

<code-tabs name="val" />

## 导出

<export-matrix />

PP-OCRv5 是一条双网络流水线，检测和识别一起走，而不是一张可追踪的计算图，导出对它没有实现：目前还不支持任何格式。如果你需要这个格式之外的检查点，就直接微调采用 Apache-2.0 许可的上游训练代码，再用 `weights/convert_ppocr_weights.py` 转换结果。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
