---
title: SwinIR
families: [swinir]
seo_title: "SwinIR：在 LibreYOLO 里做 4x 图像超分辨率"
description: "在 LibreYOLO 里用 SwinIR 做 4x 图像超分辨率。安装、预测、验证并导出轻量、中号和大号检查点。"
lead: "一个用于图像恢复的 Swin Transformer 网络。LibreYOLO 为它的 4x 超分辨率检查点提供推理和验证：官方轻量版、真实场景中号和真实场景大号生成器。"
keywords: [SwinIR, Swin Transformer, 图像超分辨率, "图像恢复 python", "swinir 4 倍放大"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSwinIRm-restore.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 分块处理大图
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile 把前向过程切成互相重叠的分块，再把接缝混合回去；tile_pad 是每个
        # 分块在被裁回去之前，四周额外加的一圈边距。两者都只是 Python 的关键字
        # 参数，不是 CLI 标志
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # 省略 imgsz 时，默认用的是一个很小的内部 patch 尺寸，而不是你的工作分
        # 辨率，所以请传入部署时真正喂给模型的尺寸
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
---

## 安装

SwinIR 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

恢复类结果不带检测框；`result.restored` 是一张稠密的 `(H, W, 3)` uint8 RGB 图
像，画布在每个维度上都是输入的 4 倍。`save=True` 会把这张图像直接写入磁盘，而不
是写一张画了标注的图。输入不会被缩放，而是被填充到 8 的倍数，所以预测跑的是照片
自身的分辨率；超出内存承受范围的数据源可以用 `tile` 和 `tile_pad` 切开，它们会在
输出里把分块的接缝混合回去。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

三个尺寸，放大倍数都固定为 4x。`s` 是官方的轻量生成器，有四个残差 Swin
Transformer 块（RSTB）阶段，配 pixel-shuffle-direct 上采样。`m` 和 `l` 是真实场
景的中号和大号生成器，分别有六个和九个 RSTB 阶段，上采样器用的是最近邻加卷积，
为真实世界的退化而设计，而不是只针对双三次下采样。

## 验证

`val()` 测量恢复输出和干净目标图像之间的 PSNR 和 SSIM，两者都在 RGB 上、按原始
画布计算，不裁边框也不做缩放。SSIM 用的是 11x11 的高斯窗口，sigma 为 1.5，在三
个颜色通道上取平均。

<code-tabs name="val" />

数据集参数是一个 YAML，把一个存放退化输入图像的目录和一个存放同分辨率干净目标
图像的目录配成对；确切的键见[数据集格式](/docs/reference/dataset-formats)。

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。ExecuTorch 以及矩阵里标为不支持的
每一种格式，这个家族都用不了；能用的是 ONNX、TorchScript、TensorRT、OpenVINO 和
TFLite。[导出](/docs/export)列出了每种格式都接受的参数，以及其中少数几种额外增加
的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
