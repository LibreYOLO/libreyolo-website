---
title: Real-ESRGAN
families: [realesrgan]
seo_title: "Real-ESRGAN：在 LibreYOLO 里做图像超分辨率"
description: "在 LibreYOLO 里用 Real-ESRGAN 做实用的图像超分辨率，覆盖 4x、2x 和一档快速 4x。安装、预测、验证并导出。"
lead: "一个实用的盲超分辨率放大模型，训练用的是合成退化，而不是只有双三次下采样。LibreYOLO 为它的 4x、2x 和快速 4x 检查点提供推理和验证。"
keywords: [Real-ESRGAN, RRDBNet, SRVGGNetCompact, 图像超分辨率, "图片放大 不失真", "图像恢复 python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRealESRGANx4-restore.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 分块处理大图
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile 把前向过程切成互相重叠的分块，再把接缝混合回去；tile_pad 是每个
        # 分块在被裁回去之前，四周额外加的一圈边距。两者都只是 Python 的关键字
        # 参数，不是 CLI 标志
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRealESRGANx4-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # 省略 imgsz 时，默认用的是一个很小的内部 patch 尺寸，而不是你的工作分
        # 辨率，所以请传入部署时真正喂给模型的尺寸
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx imgsz=512
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
---

## 安装

Real-ESRGAN 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

恢复类结果不带检测框；`result.restored` 是一张稠密的 `(H, W, 3)` uint8 RGB 图
像，画布在每个维度上都是输入的 `Results.restore_scale` 倍。`save=True` 会把这
张图像直接写入磁盘，而不是写一张画了标注的图。输入会被转成 RGB，任何 alpha 通
道都会被丢掉。超出内存承受范围的数据源可以用 `tile` 和 `tile_pad` 切开，它们会
在输出里把分块的接缝混合回去。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

三个检查点（checkpoint），按放大倍数命名。`x4` 是 RRDBNet
（`RealESRGAN_x4plus`），23 个残差中的残差稠密块，是 4x 下的画质默认选择。`x2`
是同样的 RRDBNet 架构，倍数为 2x。`x4t` 是 SRVGGNetCompact
（`realesr-general-x4v3`），一个更小、更快的生成器，为视频和 4x 下延迟更低的用
法而设计。上游的通用模型还配套发布了一个去噪强度网络，在推理时混合进来；这个强
度旋钮不在这次移植的范围内，本移植跑的是基础的 `x4t` 生成器。

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
文件的表现和检查点一样，返回同样的 `Results`。[导出](/docs/export)列出了每种格
式都接受的参数，以及其中少数几种额外增加的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
