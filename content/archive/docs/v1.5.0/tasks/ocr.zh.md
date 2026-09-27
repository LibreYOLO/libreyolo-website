---
title: 文字识别（OCR）
seo_title: OCR：LibreYOLO 中的文字检测与识别
description: 用 LibreYOLO 在图像里找到并读出文字。预测四边形和转写文本，标注一份 JSONL 数据集，并用 hmean、端到端 F1 和 1-NED 做验证。
lead: OCR 定位图像里的文字并把它读出来。LibreYOLO 把它作为 ocr 任务提供，按阅读顺序为每个文字区域返回一个四点多边形加一段转写文本。
keywords:
  - ocr python 库
  - 场景文字识别
  - ocr 中文识别
  - PP-OCRv5 python
  - 端到端文字识别
last_verified: 1.5.0
snippets:
  predict:
    - label: 读出图像里的文字
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # t 档是两档里更轻的一档，为 CPU 打造。SAMPLE_IMAGE 让这段代码可以
        # 直接跑起来，你也可以把它指向自己的一张带文字的图像
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: 读取四边形
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # (N, 4, 2) 多边形，左上 右上 右下 左下
        print(regions.xyxy)         # 这些多边形的轴对齐外接框
        print(regions.det_conf)     # 检测分数，与 .conf 分开
    - label: 按识别置信度过滤
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # 用位置索引，而不是布尔掩码：切片会把转写文本和两个分数数组
        # 连同几何一起带上
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## 定义

`ocr` 任务在一次调用里做两件事：定位图像里的每一个文字区域，并把它转写出来。区域
返回的是四点多边形而不是轴对齐的检测框，因为场景文字经常带旋转；它们按阅读顺序给
出，从上到下再从左到右。

一次预测会填充 `result.ocr`，这是一个 `OCRRegions` 载荷。`.data` 是原始图像像素
坐标下的 `(N, 4, 2)` 浮点多边形数组，四个点按左上、右上、右下、左下排列；`.texts`
是这 N 条转写文本组成的列表；`.conf` 是每个区域的识别分数，`.det_conf` 是检测分数；
`.xyxy` 给出每个多边形的轴对齐外接框。因为这些四边形是真正的多边形，它们不会填充
`result.boxes`。对 `OCRRegions` 做切片时，转写文本和两个分数数组会连同几何一起被
带上。

## 模型

有两个家族支持 `ocr`。

[PP-OCRv5](/docs/models/pp-ocrv5) 是专门的流水线：一个可微二值化（differentiable
binarization）检测器找出文字四边形，一个 SVTR/CTC 识别器把它们读出来，两个阶段连
同识别字符集一起打包在同一个 `.pt` 文件里。它分两档发布，更轻的一档面向 CPU，服务
器那一档精度更高；一部字典就覆盖简体中文、繁体中文、英文、日文和拼音。

[SenseNova-Vision](/docs/models/sensenova-vision) 做 OCR 的方式，是从支撑它另外六
个任务的同一个 7B 检查点（checkpoint）里，把这些词生成为带标签的文本，用
`LibreVLM("sensenova-vision", task="ocr")` 加载。它需要 `sensenova` extra，而且它的
权重仅限非商业用途；许可证写在它的模型页上。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

PP-OCRv5 在固定的长边上限下做检测，然后成批识别裁剪出来的区域，`rec_batch` 控制每
次前向传播送进识别器的裁剪图数量。多图数据源是顺序跑的，因为两阶段流水线不会跨图
像组批。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 数据集格式

OCR 标注是每个划分一个 JSONL 文件，每张图像一个 JSON 对象，就放在图像本身旁边。

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

每一行给出一张图像，并列出它的区域：

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` 是绝对像素坐标下的四点四边形，按左上、右上、右下、左下的顺序排列。文本
读不出来的区域标注为 `"text": "###"`，也就是 ICDAR 的 do-not-care 约定：它被排除在
识别评分之外，与它重叠的预测会被忽略，而不是算作误检。

把根目录作为 `data=` 传进去就够了。另一种做法是用数据集 YAML，写 `path` 加上可选的
`images` 和 `labels` 目录名，再用 `nc: 1` 和 `names: {0: text}` 作为 schema 占位符，
因为 OCR 模型返回的是 `Results.ocr`，而不是检测结果。完整约定见[数据集格式](/docs/reference/dataset-formats)。

## 训练

两个 OCR 家族都没有训练实现：在它们上面调用 `train()` 都会抛出
`NotImplementedError`，OCR 支持只覆盖预测和验证。PP-OCRv5 的模型页指明了采用
Apache-2.0 许可的上游训练代码，以及把微调后的检查点带回 LibreYOLO 的转换脚本。

## 验证

`val()` 对整条流水线打分，检测和识别一起算，把预测的多边形和真值（ground truth）多
边形在 IoU 高于 0.5 时一对一匹配。

<code-tabs name="val" />

`metrics/det_precision`、`metrics/det_recall` 和 `metrics/det_hmean` 只给定位打分：
一次匹配只需要多边形重叠，转写文本写了什么都不管。`metrics/e2e_precision`、
`metrics/e2e_recall` 和 `metrics/e2e_f1` 把读的部分也算进来：一次匹配需要同样的多边
形重叠，并且在 NFKC 归一化和去除空白之后转写文本完全一致，比较时区分大小写。
`metrics/e2e_f1` 同时也是 `fitness`，也就是最佳检查点选择读取的那个数。

`metrics/rec_1-NED` 只给识别器本身打分，范围是检测已经匹配上的那些配对：1 减去归一
化编辑距离，所以一条只差一个字符的转写文本能拿到接近 1 的分数，而端到端 F1 给它 0
分。

## 导出

这个任务没有可用的导出格式。PP-OCRv5 是两个网络配合着一起跑，而不是一张可追踪的计
算图，两个家族上的 `export()` 对每一种格式都会抛异常。要在 LibreYOLO 之外部署，就到
上游去微调，并走上游的部署路径。
