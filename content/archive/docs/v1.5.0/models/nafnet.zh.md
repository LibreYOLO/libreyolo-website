---
title: NAFNet
families:
  - nafnet
seo_title: NAFNet：在 MIT 许可下去噪、训练并导出
description: 在 LibreYOLO 里用 NAFNet 做图像去噪和图像恢复。安装、预测、训练、验证并导出 SIDD 检查点，采用 MIT 许可。
lead: >-
  NAFNet 是一个用于图像恢复的卷积网络，它把典型 UNet 模块里的非线性激活函数去掉，换成逐元素相乘。LibreYOLO
  支持它的一个任务，restoration，并发布了一个在 SIDD 上训练的真实图像去噪检查点。
keywords:
  - NAFNet
  - 图像恢复
  - 图像去噪 python
  - 照片降噪 深度学习
  - 图像去模糊
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: 保存恢复后的图像
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: 检查点来源
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation 和 dataset 会记录在保存的检查点上，
        # 它们不改变训练的内容
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: 多卡训练
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() 返回的是普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## 安装

NAFNet 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象为这个家族只带一个字段 `restored`，它是原始画布上一张稠密的
HWC uint8 RGB 图像；没有检测框可以遍历。`save=True` 会把这张恢复后的图像直接写入
磁盘，而不是在输入图上画一层标注。`conf`、`iou` 和 `max_det` 为了和其他每个家族
保持签名一致而被接受，但不起作用，因为图像恢复不产生任何可供过滤的检测结果。数据
源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

两种宽度共享这套架构：`s`（宽度 32）和 `l`（宽度 64），都围绕 256 px 的训练 patch
构建。无论尺寸如何，预测和验证都在原生图像分辨率下运行，只把图像补齐到网络的下采
样因子。目前只发布了 `l` 这个宽度，它是一个在 SIDD 上训练的真实图像去噪检查点
（checkpoint）。

## 训练

NAFNet 在你自己的成对退化/干净图像上微调：一个数据集 YAML，指向存放退化图像的
`inputs/<split>/` 文件夹和存放干净目标的 `targets/<split>/` 文件夹，两者按文件名主
干匹配。`degradation` 和 `dataset` 是可选字符串，记录在保存的检查点上用于溯源；它
们不参与训练。

<code-tabs name="train" />

不去动它时，训练器会跑 100 轮，用 AdamW、`lr0=1e-3`、批大小 16、256 px 裁剪，并在
PSNR 连续 50 轮没有提升后早停。这个家族没有 LoRA 路径：`lora=True` 会直接报错而不
是运行，因为 `NAFNetTrainer` 从不启用适配器微调。

训练期间网络跑的是普通的全局平均池化。NAFNet 那个只用于推理的窗口化局部池化
（Test-time Local Converter）会在第一轮开始前被摘掉，训练结束后再装回去，因为在固
定窗口的局部池化上做反向传播，和这个检查点在推理时的用法对不上。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个字典，里面有 `metrics/PSNR` 和 `metrics/SSIM`，在 RGB 上、按整张有
效画布计算：SSIM 用的是 11x11 的高斯窗口，sigma 为 1.5，而用于挑选最佳检查点的
`fitness` 就是 PSNR 值。`data` 指向的是训练时用的同一种成对图像数据集格式。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine` 文
件的表现和检查点一样，返回同样的 `Results`，由 `restored` 承载输出图像。NAFNet 以
固定的空间分辨率导出：`imgsz` 必须能被网络的下采样因子整除（两种架构宽度都是
16），而且 `dynamic=True` 时只有 batch 这一维是动态的；高和宽在导出时就固定下来。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
