---
title: PP-DocLayoutV3
families:
  - ppdoclayoutv3
seo_title: PP-DocLayoutV3 Python 用法：带阅读顺序的文档版面检测
description: >-
  在 LibreYOLO 里用 PP-DocLayoutV3 找出 25 类页面区域并按阅读顺序排序。采用 Apache-2.0 许可，纯
  PyTorch，支持导出 ONNX 和 TorchScript。
lead: >-
  PP-DocLayoutV3 是 PaddleOCR
  的文档版面检测器。它找出页面上的各类区域（正文、标题、表格、图、公式、页眉、页脚、印章等），并按阅读顺序排序。LibreYOLO 把它移植为一个
  detect 家族，不需要 transformers 或 PaddlePaddle 运行时。
keywords:
  - PP-DocLayoutV3
  - PaddleOCR
  - 文档版面分析
  - 版面检测
  - 阅读顺序
  - 版面分析 python
  - RT-DETR
  - HGNetV2
  - 文档解析
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.pt")
        result = model("page.png", conf=0.5)    # 上游流水线的阈值

        for order, box, cls, score in zip(
            result.reading_order, result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            print(int(order), result.names[int(cls)], round(float(score), 3), box.tolist())
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePPDocLayoutV3l.pt source=page.png conf=0.5
        save=True
    - label: 官方权重
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 上游的 model.safetensors 也能加载：它会被转换一次，
        # 并在旁边写出一个 LibreYOLO 检查点
        model = LibreYOLO("model.safetensors")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.pt")
        model.export(format="onnx")             # 固定 800x800，FP32
        model.export(format="torchscript")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePPDocLayoutV3l.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.onnx")
        result = model("page.png", conf=0.5)
        print(result.reading_order)
source_hash: abc633f93d73ccc5
---

## 安装

PP-DocLayoutV3 不需要任何可选 extra。它是纯 PyTorch，导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 `LibreYOLO/LibrePPDocLayoutV3l` 下载。

<code-tabs name="predict" />

`Results.boxes` 存放原始页面上的 xyxy 检测框、置信度和 25 个版面类别的类别 id。各行
按阅读顺序排序，`Results.reading_order` 是一个与检测框对齐、从 1 开始的 int64 向量。
它和 `track_id` 一样可以用 `results[i]` 切片，并以 `reading_order` 出现在
`Results.summary()` 的行里。排名是在 `conf` 过滤后保留的区域之间计算的；`classes=`
过滤或 `max_det` 截断会保留剩余区域原来的排名，所以这些值可能不连续。

上游流水线使用 `conf=0.5`；LibreYOLO 默认的 `conf=0.25` 会保留更多低置信度区域。
与上游一样，分数是逐类别的 sigmoid，保留前 300 个（query, class）对，所以同一个区域
可能带着两个标签出现。不做 NMS。

类别：`abstract`、`algorithm`、`aside_text`、`chart`、`content`、
`display_formula`、`doc_title`、`figure_title`、`footer`、`footer_image`、
`footnote`、`formula_number`、`header`、`header_image`、`image`、
`inline_formula`、`number`、`paragraph_title`、`reference`、
`reference_content`、`seal`、`table`、`text`、`vertical_text`、
`vision_footnote`。

在 `layout=True` 时，[Falcon OCR](/docs/models/falcon-ocr) 的版面阶段运行的也是
这个移植版。

## 变体

只有一个尺寸，`l`：一个 RT-DETR 风格的集合预测检测器，采用 HGNetV2-L 骨干、掩码增强
的 query 初始化，以及一个为每对区域打分的 global-pointer head。输入是固定的 800x800
画布（拉伸缩放，RGB 取值 `[0, 1]`，不做归一化）。它可以在 CPU、CUDA 和 MPS 上运行；
在 MPS 上位置编码以 float32 计算，所以结果在最后几位上可能不同。

与 transformers 里采用 Apache-2.0 许可的 PyTorch 参考实现相比，在 CUDA 上以 fp32、
相同输入张量运行时，logits、检测框、阅读顺序 logits 和掩码在三张真实页面和一个随机
张量上完全一致。不提供训练：训练需要阅读顺序和逐区域掩码的监督信号，而 LibreYOLO 的
detect 数据集不带这些。上游从掩码分支推导出的逐区域多边形没有暴露出来。

## 导出

<export-matrix />

ONNX（opset 17）和 TorchScript，固定 800x800，FP32。计算图输出 `pred_logits`、
`pred_boxes` 和 `order_votes`；导出后的后端返回同样的 `Results`，包括
`reading_order`。

<code-tabs name="export" />

## 检查点

<checkpoint-table />

## 许可证

<provenance-box>

PP-DocLayoutV3 的代码和权重都采用 Apache-2.0 许可，出自
[PaddlePaddle Authors](https://github.com/PaddlePaddle/PaddleOCR)。允许商用。
检查点（checkpoint）由 `PaddlePaddle/PP-DocLayoutV3_safetensors` 转换而来，只去掉了
一个键前缀并添加了元数据；学到的参数没有改动，镜像不需要申请访问。

</provenance-box>

## 引用

<citation-block />
