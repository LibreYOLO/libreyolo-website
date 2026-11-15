---
title: PRX
families:
  - prx
seo_title: PRX 与 PRX Pixel Python 用法：Photoroom 的开源文生图模型
description: >-
  在 LibreYOLO 里通过 LibreGen 运行 Photoroom 的 PRX 和 PRX Pixel 文生图模型。九个 256、512 和
  1024 分辨率的 PRX 检查点，外加 7B 的 PRX Pixel。采用 Apache-2.0 许可。
lead: >-
  PRX 是 Photoroom 的开源文生图模型，一个 1.3B 的 transformer，以 256、512 和 1024 px
  共九个检查点发布。PRX Pixel 是它的 7B 兄弟模型，直接在像素上去噪，没有自编码器。LibreYOLO 通过 LibreGen 工厂，在
  diffusers 流水线上运行全部十个模型。
keywords:
  - PRX
  - PRX Pixel
  - PRXPixel
  - Photoroom
  - 文生图 python
  - 图像生成
  - 扩散模型 diffusion
  - DC-AE
  - T5Gemma
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("prx-512-sft")
        result = model("a red fox sleeping in fresh snow", seed=0)
        result.save("fox.png")             # PNG 里还会保存 prompt、seed 和参数
    - label: 设置
      language: python
      code: >
        from libreyolo import LibreGen


        model = LibreGen("prx-512-sft-distilled")          # 8 步蒸馏版

        result = model("a red fox", steps=8, guidance=1.0)


        model = LibreGen("prx")                            # prx-1024-beta

        result = model("a red fox", negative="blurry", imgsz=1024, seed=0)


        model = LibreGen("prxpixel", offload=True)         # prxpixel-7b，需要
        diffusers>=0.39

        results = model(["a red fox", "a blue jay"], n=2, save=True)
    - label: CLI
      language: bash
      code: >
        libreyolo generate prompt="a red fox" model=prx-512-sft
        negative="blurry"

        libreyolo generate prompt="a red fox" model=prxpixel-7b seed=0 n=4
source_hash: 5eeb7c71d04d09f8
---

## 安装

```bash
pip install "libreyolo[gen]"
```

PRX Pixel 需要 `diffusers>=0.39`，加载时会给出提示；这个 extra 的下限是 0.38，这样
它可以和 Marigold 的 extra 共存。

## 生成

权重在首次使用时从 LibreYOLO 的镜像下载，这些镜像与 Photoroom 的仓库逐字节一致。

<code-tabs name="predict" />

字符串永远是提示词。一个提示词配 `n=1` 返回一个 `Results`；提示词列表或 `n > 1`
返回一个扁平列表，`stream=True` 返回一个生成器。`result.generated` 存放图像
（`array`、`image`）、`prompt`、能复现它的 `seed`，以及生效的 `params`。每个家族
参数都是可选的；传入家族没有的参数会抛出异常并列出支持的参数。提示词、seed、设备和
offload 见[图像生成](/docs/tasks/image-generation)任务页。

## 变体

| 模型 | 参数量 | 可调参数（默认值） | GPU 显存（库给出的数字） |
| --- | --- | --- | --- |
| `prx-1024-beta`（别名 `prx`） | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 约 21 GB |
| `prx-512-sft-distilled` | 1.3B | `steps`（8）、`guidance`（1.0）、`negative`（""） | 约 16 GB |
| `prx-512-sft`、`prx-512` | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 约 16 GB |
| `prx-512-dc-ae-sft-distilled` | 1.3B | `steps`（8）、`guidance`（1.0）、`negative`（""） | 约 16 GB |
| `prx-512-dc-ae-sft`、`prx-512-dc-ae` | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 约 16 GB |
| `prx-256-sft`、`prx-256` | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 约 16 GB |
| `prxpixel-7b`（别名 `prxpixel`） | 7B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 约 18 GB |

数字是原生分辨率。`-dc-ae` 用 32x 的 DC-AE 自编码器代替 FLUX.1 VAE，所以边长向上取整
到 32 的倍数，而不是 16。`-sft` 是监督微调版，可以直接使用，`-sft-distilled` 是它的
8 步蒸馏版。不带 `-sft` 的检查点是用于微调的基础模型。`prx-1024-beta` 是 Photoroom 的
1024 预览版。PRX Pixel 自带一个 Qwen3-VL 文本编码器，没有自编码器。

## 导出

<export-matrix />

与所有 LibreGen 家族一样，`export()`、`train()` 和 `val()` 会抛出异常。

## 许可证

<provenance-box>

PRX 和 PRX Pixel 出自 [Photoroom](https://github.com/Photoroom/PRX)。权重采用
Apache-2.0 许可，所以可以商用。例外是 PRX 自带的 T5Gemma 2B 文本编码器，它仍受
Google 的 **Gemma Terms of Use** 和 Prohibited Use Policy 约束：允许商用，但禁止的
用途会传递给你分享给的任何人，而且 PRX 的模型卡对整个模型禁止这些用途。自编码器是
FLUX.1 [schnell] VAE（Apache-2.0）或 DC-AE（MIT）；PRX Pixel 的
Qwen3-VL-Embedding-2B 文本编码器采用 Apache-2.0 许可。

LibreYOLO 的适配器是它自己采用 MIT 许可的代码，构建在采用 Apache-2.0 许可的 diffusers
`PRXPipeline` 和 `PRXPixelPipeline` 之上。权重从 `LibreYOLO/LibrePRX<size>` 和
`LibreYOLO/LibrePRXPixel7b` 下载，这些是固定到某个 commit、逐字节一致、不需要申请
访问的镜像。

</provenance-box>

## 引用

<citation-block />
