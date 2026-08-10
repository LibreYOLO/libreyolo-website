---
title: TEED
families: [teed]
seo_title: "TEED：边缘检测，自备检查点"
description: "在 LibreYOLO 里用 TEED 做稠密的边缘概率预测。转换一个你持有许可的检查点，然后预测、验证并导出它。"
lead: "TEED（Tiny and Efficient Edge Detector）是一个小型卷积网络，从一张 RGB 图像预测出稠密的边缘概率图。LibreYOLO 只为边缘检测封装了它的架构；库里不附带任何检查点。"
keywords: [TEED, Tiny and Efficient Edge Detector, "边缘检测 python", BIPED, "图像边缘检测 深度学习"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32，取值在 [0, 1]
        print(edges.binary(0.5).sum())  # 阈值化之后的边缘像素数
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=weights/LibreTEEDt-edge.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # 数据集最优尺度下的 F-measure
        print(metrics["metrics/OIS"])   # 单图最优尺度下的 F-measure
    - label: CLI
      language: bash
      code: |
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352
        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt imgsz=352 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
---

## 安装

TEED 不需要任何可选的额外依赖。它导入的所有东西都在基础安装里。

```bash
pip install libreyolo
```

## 预测

LibreYOLO 不发布任何 TEED 检查点（checkpoint）。官方发布的权重是在 BIPED 上
训练的，而 BIPED 公布的数据集条款把用途限制在非商业目的，所以 LibreYOLO 不镜像
它们。用 `weights/convert_teed_weights.py` 转换一个你有权使用的检查点，它会
先把张量的键和运行时架构对照检查一遍，再写出一个 LibreYOLO 可以直接加载的文件：

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` 保存着结果：一个取值在 `[0, 1]` 之间的 `(H, W)` float32 数组，
`.binary(threshold)` 会返回一张布尔边缘掩码。这里没有检测框，所以 `conf`、`iou`
和 `max_det` 都不起作用。输入源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

TEED 在 LibreYOLO 里只有一个尺寸。LibreYOLO 的基准测试工具还没有测过这个家族，
所以没有可以拿来对比的公开数字。

## 验证

`val()` 会在成对的边缘数据集上报告 BSDS 风格的 ODS 和 OIS F-measure：图片和同名的
边缘图放在一起，还可以带一张可选的有效性掩码，让填充像素永远不计入。`imgsz` 必须
能被网络的下采样步长整除，不满足时 LibreYOLO 会抛出一个明确的错误。

<code-tabs name="val" />

## 导出

<export-matrix />

边缘导出使用固定分辨率、batch 为 1 的运行时约定：`dynamic` 以及 1 以外的 `batch`
都会被拒绝，导出的图只输出一张融合后的概率图。导出产物按文件后缀通过
`LibreYOLO()` 加载回来，所以一个 `.onnx` 文件的表现和检查点一样，返回同样的
`Results`。

<code-tabs name="export" />

## 许可

<provenance-box>

LibreYOLO 不发布任何 TEED 检查点。LibreYOLO 组织下没有镜像任何东西；请改用
`weights/convert_teed_weights.py` 转换一个你持有许可的检查点。

</provenance-box>

## 引用

<citation-block />
