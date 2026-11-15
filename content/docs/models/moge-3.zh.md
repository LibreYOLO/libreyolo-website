---
title: MoGe-3
families:
  - moge3
seo_title: MoGe-3 Python 用法：从一张图像得到度量深度、点图和相机参数
description: >-
  在 LibreYOLO 里用 MoGe-3 从一张 RGB 图像得到度量深度、相机坐标系点图、相机内参和表面法线。提供 ViT-L 和 ViT-g，采用
  MIT 许可。
lead: >-
  MoGe-3 是微软（Microsoft）的单目几何模型。它从一张 RGB
  图像预测以米为单位的度量深度、相机坐标系下的点图、相机内参、有效性掩码和表面法线。它是 MoGe-2 加上一个稀疏 3D
  UNet，后者在体素网格上细化预测的深度。LibreYOLO 提供官方的 ViT-L 和 ViT-g 检查点，用于推理、验证和法线导出。
keywords:
  - MoGe-3
  - MoGe 3
  - 度量深度估计
  - 单目几何
  - 点图 point map
  - 相机内参
  - 表面法线
  - moge python
  - DINOv2
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe3l-depth.pt")
        depth = model(SAMPLE_IMAGE).depth_map

        print(depth.data.shape)        # (H, W)，单位为米，有效掩码之外为 inf
        print(depth.point_map.shape)   # (H, W, 3)，相机坐标系下的点，单位为米
        print(depth.intrinsics)        # (3, 3)，OpenCV 像素内参
        print(depth.valid_mask.shape)  # (H, W) bool
    - label: 法线
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe3l-normal.pt")
        normal = model(SAMPLE_IMAGE).normal_map
        print(normal.array.shape)      # (H, W, 3) 单位向量，OpenCV 坐标系
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        save=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 只有法线计算图能导出；深度导出被阻止
        model = LibreYOLO("LibreMoGe3l-normal.pt")
        model.export(format="onnx")
        model.export(format="torchscript")
source_hash: f9d2d8df5141447a
---

## 安装

MoGe-3 不需要任何可选 extra。稀疏细化器是纯 PyTorch，相机参数恢复用的是 SciPy，
它在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 镜像下载。

<code-tabs name="predict" />

使用 `-depth` 检查点时，`result.depth_map` 存放以米为单位的度量深度
（`encoding="depth"`、`metric=True`）。点图、内参和有效性掩码作为
`depth_map.point_map`、`depth_map.intrinsics` 和 `depth_map.valid_mask` 附在它
上面；掩码之外的像素为 `inf`。点使用 OpenCV 坐标轴，内参把像素中心放在整数坐标上。
使用 `-normal` 检查点时，`result.normal_map` 存放 OpenCV 相机坐标系下的单位法线。

`imgsz` 设置 ViT 的 token 预算，即按图像宽高比排布的 `(imgsz // 14) ** 2` 个 token。
默认值 840 是上游的最高分辨率档（3600 个 token）；490 接近它的最低档。输入保持原生
分辨率，每个输出都会插值回这个分辨率。推理以 FP32 运行，并且像上游默认做法一样总是
执行三步细化。每次前向传播处理一张图像；没有堆叠批处理的路径。

CPU、CUDA 和 MPS 都能运行，不需要编译扩展。细化器的内存随 `imgsz` 增长，而不是随
输入分辨率增长。

## 变体

| 权重文件 | 骨干 | 任务 |
| --- | --- | --- |
| `LibreMoGe3l-depth.pt` | DINOv2 ViT-L/14 | 深度、点图、内参、掩码 |
| `LibreMoGe3l-normal.pt` | DINOv2 ViT-L/14 | 法线 |
| `LibreMoGe3g-depth.pt` | DINOv2 ViT-g/14 | 深度、点图、内参、掩码 |
| `LibreMoGe3g-normal.pt` | DINOv2 ViT-g/14 | 法线 |

同一尺寸的 `-depth` 和 `-normal` 文件存放的是相同的张量；后缀决定任务。在 RTX 3090
上以 FP32 与上游对比，单次前向的计算图逐位一致，细化后的深度一致到浮点舍入误差
（`l` 的 p99 相对差异为 4.7e-6，`g` 为 3.7e-5）。

MoGe-3 在 LibreYOLO 里不可训练：`train()` 会抛出异常。零样本 `val()` 运行共用的
深度和法线验证器。

## 导出

<export-matrix />

只有法线计算图能导出，支持 ONNX 和 TorchScript，使用固定的正方形画布。深度导出被
阻止：细化器构建的体素集合依赖数据，而相机参数恢复是在 CPU 上求解的。

<code-tabs name="export" />

## 检查点

<checkpoint-table />

## 许可证

<provenance-box>

MoGe-3 出自微软（[microsoft/MoGe](https://github.com/microsoft/MoGe)）。代码和
权重采用 MIT 许可，所以可以商用。DINOv2 骨干保留其 Apache-2.0 署名。LibreYOLO 把
官方的 `Ruicheng/moge-3-vitl` 和 `Ruicheng/moge-3-vitg` 检查点（checkpoint）原样
镜像为 `LibreYOLO/LibreMoGe3{l,g}-{depth,normal}`，不需要申请访问。

</provenance-box>

## 引用

<citation-block />
