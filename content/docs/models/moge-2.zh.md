---
title: MoGe-2
families: [moge2]
seo_title: "MoGe-2：预测、验证并导出表面法线"
description: "在 LibreYOLO 里用 MoGe-2 做稠密表面法线预测。安装、预测、验证并导出官方的 ViT-S、ViT-B 和 ViT-L 检查点。"
lead: "MoGe-2 是一个单次前向的单目几何模型，从一张 RGB 图像预测稠密的表面法线场。LibreYOLO 只把它用于法线估计，通过官方的 ViT-S、ViT-B 和 ViT-L 检查点。"
keywords: [MoGe-2, MoGe 2, "表面法线估计", "法线图 python", "单目几何", "稠密预测", DINOv2]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3) float32 单位向量
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMoGe2s-normal.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # 单位为度
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # 像素百分比
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518
        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
---

## 安装

MoGe-2 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时自动下载：LibreYOLO 直接从官方检查点获取对应的尺寸，并缓存在本地。

<code-tabs name="predict" />

MoGe-2 返回的是一个稠密场，而不是一组检测结果，所以 `result.boxes` 是空的，
`conf`、`iou` 和 `max_det` 都不起作用。`result.normal_map` 装的是结果：一个
`(H, W, 3)` 的单位向量数组，位于 OpenCV 相机坐标系中，其中 `+x` 向右，`+y` 向下，
`+z` 指向场景内部，正对相机的表面读作 `(0, 0, -1)`。对一个图像列表做预测时，每张
图像跑一次前向；这个家族没有堆叠成批的快速路径。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

三种编码器尺寸以独立的检查点（checkpoint）发布：ViT-S、ViT-B 和 ViT-L，输入
分辨率都相同。LibreYOLO 的基准测试工具尚未测过这个家族，所以没有公开的精度数字
可供比较；请按你自己的算力预算挑一个尺寸。

## 验证

`val()` 在成对的法线图数据集上测量角度误差：图像旁边放文件名主干相同的 16 位法线
PNG，还可以带一个可选的有效性掩码，让填充像素和无效像素都不计入。它返回以度为
单位的平均角度误差和中位角度误差，外加落在 11.25、22.5 和 30 度以内的像素百分比。

<code-tabs name="val" />

## 导出

<export-matrix />

法线导出使用固定分辨率、批大小为 1 的运行时契约：`dynamic` 以及 1 以外的 `batch`
都会被拒绝，而且 `imgsz` 必须能被 ViT 编码器的 patch size 整除，LibreYOLO 会在
运行开始前检查这一点。导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个
`.onnx` 文件的表现和检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

## 许可证

<provenance-box>

LibreYOLO 不会把这些检查点复制到自己的组织下。`LibreYOLO("LibreMoGe2s-normal.pt")`
会按固定的修订版本（revision）直接从官方 Hugging Face 仓库下载对应的尺寸，并在
使用前对照记录的 SHA-256 校验和验证该文件。

</provenance-box>

## 引用

<citation-block />
