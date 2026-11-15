---
title: RiO-DETRv2
families:
  - riodetrv2
seo_title: RiO-DETRv2 Python 用法：实时旋转框目标检测，采用 Apache-2.0 许可
description: >-
  在 LibreYOLO 里用 RiO-DETRv2 在航拍图像上做旋转框检测。提供 n 到 x 的尺寸，以及 DOTA-v1.0 和 DIOR-R
  检查点，代码和权重均采用 Apache-2.0 许可。
lead: >-
  RiO-DETRv2 是一个实时的旋转框
  DETR。它在框自身的坐标系里把框的四条边预测为分布，由粗到细地细化角度，并用一个感知角度的质量分数为每个框打分。LibreYOLO
  以仅推理的方式移植了它，提供五种尺寸，以及 DOTA-v1.0 和 DIOR-R 检查点。
keywords:
  - RiO-DETRv2
  - RiO-DETR
  - 旋转目标检测
  - OBB 旋转框
  - 旋转框检测 python
  - DOTA
  - DIOR-R
  - 遥感 航拍图像
  - DETR
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb.pt")        # DOTA-v1.0，15 个类别
        result = model("aerial.jpg", save=True)

        print(result.obb.xywhr)   # (N, 5)：中心、尺寸和角度
        print(result.obb.conf, result.obb.cls)
    - label: DIOR-R
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb-diorr.pt")  # DIOR-R，20 个类别
        result = model("aerial.jpg")
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRiODETRv2s-obb.pt source=aerial.jpg
        save=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb.pt")
        model.export(format="onnx")
        model.export(format="torchscript")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRiODETRv2s-obb.pt format=onnx
source_hash: a964bd82aed1c2bc
---

## 安装

RiO-DETRv2 不需要任何可选 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 镜像下载。这些模型是在航拍图像上训练的，
所以请使用俯拍的照片。

<code-tabs name="predict" />

`result.obb` 存放旋转框：`xywhr`（中心、宽、高和角度）、`conf` 和 `cls`，类别名来自
检查点。载荷和绘图见[旋转框检测](/docs/tasks/oriented-detection)任务页。

## 变体

五种尺寸 `n`、`s`、`m`、`l` 和 `x`，每种有两个检查点：

| 检查点 | 数据集 | 类别数 | 画布 |
| --- | --- | ---: | ---: |
| `LibreRiODETRv2<size>-obb.pt` | DOTA-v1.0 | 15 | 1024 px |
| `LibreRiODETRv2<size>-obb-diorr.pt` | DIOR-R | 20 | 800 px |

原生画布尺寸从检查点里读取。在 CPU 上，全部十个检查点的原始输出和解码后的旋转框都与
上游采用 Apache-2.0 许可的发布版完全一致（最大绝对差为 0）。

RiO-DETRv2 仅支持推理。`train()` 会抛出异常：上游的训练配方（DINOv3 特征蒸馏、OBB
匈牙利匹配、边界和角度分布损失函数）没有移植。需要可训练的旋转框检测，请用
[GTR](/docs/models/gtr) 或 [YOLO-NAS](/docs/models/yolo-nas)。

## 导出

<export-matrix />

ONNX 和 TorchScript 导出可以往返：导出的文件能通过 `LibreYOLO()` 重新加载，并返回
同样的 `Results`。

<code-tabs name="export" />

## 检查点

<checkpoint-table />

## 许可证

<provenance-box>

RiO-DETRv2 出自 RicePasteM
（[RicePasteM/RiO-DETRv2](https://github.com/RicePasteM/RiO-DETRv2)）。代码和
权重采用 Apache-2.0 许可，所以可以商用。上游在首次提交约 33 分钟后把 LICENSE 文件
改成了 AGPL-3.0，代码没有任何改动。LibreYOLO 的移植只使用最初那个采用 Apache-2.0
许可的版本（commit `ab8b22d`），并在 `LibreYOLO/RiO-DETRv2` 保留了这份快照。官方
检查点（checkpoint）镜像为 `LibreYOLO/LibreRiODETRv2{n,s,m,l,x}-obb` 和
`-obb-diorr`，只重命名了键，学到的张量没有改动。DOTA-v1.0 和 DIOR-R 训练集有各自的
条款。

</provenance-box>

## 引用

<citation-block />
