---
title: 图像修复
seo_title: LibreYOLO 中的图像修复与超分辨率
description: 在 LibreYOLO 中做图像去噪、去模糊和超分辨率。预测修复后的 RGB 图像，在成对数据上训练 NAFNet，并读取 PSNR 与 SSIM 指标键。
lead: >-
  图像修复接收一张退化的图像，返回一张干净的图像。LibreYOLO 把它开放为 restore 任务，在同一份输出契约下涵盖去噪、去模糊和超分辨率：输入一张
  RGB 图像，输出一张 RGB 图像。
keywords:
  - 图像修复 python
  - 图像去噪模型
  - 图像超分辨率 python
  - 图像去模糊模型
  - PSNR SSIM 评价指标
last_verified: 1.5.0
snippets:
  predict:
    - label: 放大一张图像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 紧凑的 4x 生成器，tile 可以限制大图上的峰值内存
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # 每个轴都是输入的 4 倍
    - label: 对一张图像去噪
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 在 SIDD 真实图像噪声上训练，输出保持输入尺寸
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
        print(result.restore_scale)   # 1：该检查点不做放大
  train:
    - label: 在成对图像上微调 NAFNet
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: 把来源信息记录到检查点里
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation 和 dataset 会写入保存的检查点，用于记录来源，
        # 它们不参与训练
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() 返回的是普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: 导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz 会固化进计算图，所以要传入你部署时
        # 真正喂给模型的尺寸
        model.export(format="onnx", imgsz=256)
    - label: 运行导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物能像普通检查点一样加载，
        # 并返回同样的 Results 对象
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## 定义

`restore` 任务把一张图像映射为另一张图像。去噪、去模糊和超分辨率在这里都是同一个
任务，因为它们共用同一份契约：模型接收一张 RGB 图像并返回一张 RGB 图像，而它被训练
去消除的退化类型是检查点（checkpoint）的属性，而不是 API 的属性。

一次预测会填充 `result.restored`，这是一个 `RestoredImage` 载荷，里面是一个
`(H, W, 3)` 的 uint8 RGB 数组。`.array` 以 NumPy 形式返回它，`.save(path)` 把它写入
磁盘。`result.restore_scale` 记录输出画布携带的放大倍数，对于保持分辨率的检查点它
是 `1`。`result.boxes` 始终为空，所以 `conf`、`iou` 和 `max_det` 只是为了签名一致才
被接受，实际不起作用，而 `save=True` 直接写出修复后的图像，而不是一张带标注的照片。

## 模型

有三个家族服务于 `restore`，按它们消除的退化类型划分。

[NAFNet](/docs/models/nafnet) 是去噪模型，也是 LibreYOLO 唯一能训练的修复家族。它的
架构把 UNet 块里的非线性激活换成了逐元素相乘，公开的检查点在 SIDD 真实图像噪声上
训练。输出保持输入分辨率。

[Real-ESRGAN](/docs/models/real-esrgan) 是实用的放大模型：三个检查点针对合成退化
训练，而不只是双三次下采样，分别是 4x、2x，以及一个更小更快、为更低延迟设计的 4x
生成器。

[SwinIR](/docs/models/swinir) 用 Swin Transformer 骨干做 4x 放大，提供三种尺寸，
覆盖官方的轻量生成器和两个面向真实场景的生成器。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存到本地。

<code-tabs name="predict" />

修复是在源图像自身的分辨率上运行的，而不是固定的网络画布，只补齐到网络的下采样
倍数，所以时间和内存都随输入的像素数增长。`tile` 把前向过程拆成互相重叠的小块，
再把接缝混合回去，`tile_pad` 是每个小块被裁回之前在四周加上的余量；两者都是 Python
关键字参数。关于输入源、流式处理和结果处理，见[预测](/docs/predict)。

## 数据集格式

修复把每张退化的输入图像与一张分辨率完全相同的干净目标图像配对，按文件名主干匹配。

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` 和 `names` 是 schema 占位符，修复模型返回的是 `Results.restored`，不是检测
结果。`degradation` 和 `dataset` 是可选的来源标签。`target_stem_suffix` 用于干净图像
与其退化配对命名方式不同的数据集。验证保持原生分辨率，只补齐到刚好能把一个批堆叠
起来，所以指标是在原始画布上计算的。完整契约见[数据集格式](/docs/reference/dataset-formats)。

## 训练

NAFNet 是唯一带训练实现的修复家族。`Real-ESRGAN.train()` 和 `SwinIR.train()` 都会
抛出 `NotImplementedError`：那些检查点来自在合成退化流水线上做的 GAN 训练，而成对
修复训练器只会照常跑完，并不会复现那套配方。

<code-tabs name="train" />

训练器对输入和目标这一对图像取耦合的裁剪块，所以两边始终对齐。关于数据集、多卡
训练和日志记录器，见[训练](/docs/train)；关于这个家族的默认值，以及它在训练时切断的
推理期池化，见 [NAFNet 页面](/docs/models/nafnet)。

## 验证

`val()` 在 RGB 空间、在原始画布上把修复输出与干净目标做比较，不裁边框，也不缩放。

<code-tabs name="val" />

`metrics/PSNR` 是以分贝为单位的峰值信噪比，它同时也是 `fitness`，即挑选最佳检查点
时读取的那个数。`metrics/SSIM` 是取值在 `[0, 1]` 的结构相似度，用 sigma 为 1.5 的
11x11 高斯窗计算，并在三个颜色通道上取平均。两者都是越高越好。

## 导出

导出的修复模型按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点一样，返回同样的 `Results`，输出图像由 `restored` 携带。

<code-tabs name="export" />

修复模型的导出会把空间分辨率固化进计算图，所以要传入你部署时真正喂给模型的
`imgsz`。对 NAFNet 来说，这个尺寸必须能被网络的下采样倍数整除，而在 `dynamic=True`
下只有批维度保持动态。对 Real-ESRGAN 和 SwinIR，不传 `imgsz` 会回退到一个较小的
内部 patch 尺寸，而不是你实际的工作分辨率。各格式的覆盖情况见每个模型页面和
[完整导出矩阵](/docs/reference/export-matrix)。[导出](/docs/export)列出了每种格式
接受的参数。
