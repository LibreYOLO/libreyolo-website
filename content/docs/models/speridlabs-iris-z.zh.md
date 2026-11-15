---
title: Speridlabs Iris-Z
families:
  - speridlabs_irisz
seo_title: Speridlabs Iris-Z Python 用法：一个模型同时做图像和深度
description: >-
  在 LibreYOLO 里运行 Speridlabs Iris-Z：用同一套基于 FLUX.2
  的权重完成文生图、图像转深度、深度转图像，以及同时生成图像和深度。权重仅限非商用。
lead: >-
  Speridlabs Iris-Z 是在 FLUX.2 上微调、在同一个序列里对图像及其深度一起去噪的模型。同一套权重可以根据提示词生成图像、用 4
  步从照片估计深度、在给定深度上按提示词作画，以及同时返回一张图像和它的深度。LibreYOLO 通过 LibreGen 工厂运行它的两个尺寸。
keywords:
  - Speridlabs Iris-Z
  - Iris-Z
  - Speridlabs
  - FLUX.2
  - 深度估计
  - 文生图
  - 深度转图像
  - RGB-D 生成
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: 图像转深度
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("speridlabs-iris-z", task="depth")   # 9b；不加载文本编码器
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        print(depth.encoding)          # "log_depth"：值越大越远
        print(depth.data.shape)        # (H, W)，与照片尺寸相同
        print(depth.valid_mask.shape)  # 天空处为 False
        result.save("depth.png")       # 上色后的深度图
    - label: 文生图
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        result = model("a lighthouse at dusk", seed=0)
        result.generated.save("lighthouse.png")
    - label: 深度转图像
      language: python
      code: >
        from libreyolo import LibreGen, SAMPLE_IMAGE


        model = LibreGen("speridlabs-iris-z")

        # 先估计照片的深度，再在上面画出一个新场景

        result = model("a sunken ruin under the sea", source=SAMPLE_IMAGE,
        seed=0)

        result.generated.save("ruin.png")
    - label: 图像加深度
      language: python
      code: >
        from libreyolo import LibreGen


        model = LibreGen("speridlabs-iris-z")

        result = model("a grand old library, a spiral staircase", depth=True,
        seed=0)

        result.generated.save("library.png")

        print(result.depth_map.data.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo generate prompt="a lighthouse at dusk" model=speridlabs-iris-z

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        task=depth

        libreyolo generate source=photos/ task=depth ensemble=10 imgsz=1024
source_hash: c54a3e5969e7642e
---

## 安装

```bash
pip install "libreyolo[gen]"
```

## 预测

任务在模型加载时就确定了：图像转深度用 `task="depth"`，其他三种模式用默认的
`generate`。深度模型从不加载文本编码器。

<code-tabs name="predict" />

**图像转深度。** 数据源与 predict 接受的相同。`result.depth_map` 是一个
`DepthMap`，`encoding="log_depth"`（相对对数深度，值越大越远），尺寸与数据源相同，
`valid_mask` 在天空处为 `False`。`imgsz` 是工作分辨率，512 或 1024（不到五十万像素
时默认 512，否则 1024）。可调参数：`steps`（4）和 `ensemble`（1），后者是在那么多个
seed 上取逐像素中位数。默认 seed 为 0，所以深度结果可以复现。`save=True` 把上色后的
深度图写到 `runs/depth/predict` 下。

**文生图。** 提示词、`imgsz`、`seed` 和 `n` 的用法与所有
[图像生成](/docs/tasks/image-generation)模型相同。可调参数：`steps`（50）和
`guidance`（`9b` 为 4.0，`32b` 为 3.5）。

**深度转图像。** generate 任务上的 `source=` 是作画所依据的深度：可以是一张照片
（先估计它的深度）、任意深度模型给出的 `Results` 或 `DepthMap`，或者配合
`depth_format="depth"`（越亮越远）或 `"disparity"`（越亮越近）的灰度深度图。只有当
`depth_format` 这样指定时，图像才会被当作深度图读取。输出保持深度的宽高比，
`Results.depth_map` 就是作画所依据的深度。

**图像加深度。** `depth=True` 联合采样图像和深度，然后像上游默认做法一样，根据生成
好的图像重新估计深度。结果同时带有 `result.generated` 和 `result.depth_map`。

CLI 支持文生图和图像转深度；在 CLI 里 `prompt=` 和 `source=` 仍然互斥。

## 变体

| 尺寸 | 别名 | 参数量 | 文本编码器 | GPU 显存（库给出的数字） |
| --- | --- | --- | --- | --- |
| `9b` | `speridlabs-iris-z`、`-9b`、`-small` | 9.1B，基于 FLUX.2 [klein] Base 9B | Qwen3 8B FP8 | 约 18 GB，另加 9 GB 文本编码器和 0.2 GB VAE |
| `32b` | `speridlabs-iris-z-32b`、`-big` | 32.2B，基于 FLUX.2 [dev] | Mistral Small 3.2 24B | 约 64 GB，另加 48 GB 文本编码器和 0.2 GB VAE |

`offload=True` 会在两次提示之间把文本编码器留在 CPU 上。在缺少 transformers 所需
FP8 内核（`kernels` 包）的 CUDA 上，以及在 MPS 或 CPU 上，`9b` 的文本编码器会反量化
为 bfloat16 运行，结果接近但不完全等同于上游的嵌入向量。本地快照或变体文件夹可以用
`LibreGen(path)` 加载。

## 导出

<export-matrix />

与所有 LibreGen 家族一样，`export()`、`train()` 和 `val()` 会抛出异常。

## 许可证

<provenance-box>

Speridlabs Iris-Z 出自 [Speridlabs](https://speridlabs.com/research/iris-z)。
权重采用 Black Forest Labs 的 **FLUX Non-Commercial License**：不得用于商业或生产
用途，第 2(e) 条要求在分享生成图像之前进行内容过滤或人工审核。LibreYOLO 的适配器
采用 MIT 许可；从 [Iris-Z SDK](https://github.com/speridlabs/iris-z-sdk) 移植的
transformer、采样器和提示词嵌入器保留 Apache-2.0 许可。

LibreYOLO 不再分发任何权重。它们在运行时从 `speridlabs/iris-z` 的固定 revision
原样下载，不需要申请访问。FLUX.2 VAE 来自采用 Apache-2.0 许可的
`black-forest-labs/FLUX.2-klein-4B` 仓库，文本编码器（`9b` 用 `Qwen/Qwen3-8B-FP8`，
`32b` 用 `mistralai/Mistral-Small-3.2-24B-Instruct-2506`）采用 Apache-2.0 许可。

</provenance-box>

## 引用

<citation-block />
