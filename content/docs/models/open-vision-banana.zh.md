---
title: Open Vision Banana
families:
  - open_vision_banana
seo_title: Open Vision Banana Python 用法：用提示词做分割、深度和法线
description: >-
  在 LibreYOLO 里运行 Open Vision Banana：一个用图像回答感知任务的 FLUX.2 [klein]
  微调模型。支持分割、深度、法线、图像编辑和文生图。权重仅限非商用。
lead: >-
  Open Vision Banana（OVB）出自 FunAILab，是在 FLUX.2 [klein] Base
  上微调、用图像回答感知任务的模型。提示词指定任务，输出图像编码答案：分割是平涂的颜色，深度是灰度，表面法线是
  RGB。同一套权重仍保留图像编辑和文生图能力。LibreYOLO 通过 LibreGen 工厂运行它的 4B 和 9B 版本。
keywords:
  - Open Vision Banana
  - OVB
  - FunAILab
  - FLUX.2 klein
  - 生成式感知
  - 语义分割
  - 深度估计
  - 表面法线
  - 图像编辑
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: 深度
      language: python
      code: >
        from libreyolo import LibreGen, SAMPLE_IMAGE


        model = LibreGen("open-vision-banana")      # 4b；9b 用
        "open-vision-banana-9b"

        DEPTH = (
            "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
            "for every input pixel at the same pixel location, with closer points brighter and farther points "
            "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
            "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
            "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
            "legends, borders, or objects; return only the registered depth map."
        )

        result = model(DEPTH, source=SAMPLE_IMAGE, seed=0)

        result.generated.save("depth.png")

        rgb = result.generated.array                # (H, W, 3) uint8
    - label: 语义分割
      language: python
      code: >
        import json

        from libreyolo import LibreGen, SAMPLE_IMAGE


        model = LibreGen("open-vision-banana")

        legend = {"building": [255, 8, 232], "sky": [2, 204, 254], "water": [8,
        4, 249]}

        prompt = (
            "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
            f"using this color mapping: {json.dumps(legend, separators=(', ', ': '))} "
            "Color each pixel of a listed class with its color and leave everything else black. "
            "Output only the segmentation visualization image."
        )

        model(prompt, source=SAMPLE_IMAGE,
        seed=0).generated.save("segmentation.png")
    - label: 文生图
      language: python
      code: >
        from libreyolo import LibreGen


        model = LibreGen("open-vision-banana")

        model("an empty tram at a cobblestone stop on a rainy night",
        imgsz=1024, seed=0).generated.save("tram.png")
    - label: CLI
      language: bash
      code: >
        # CLI 只支持文生图

        libreyolo generate model=open-vision-banana prompt="an empty tram at a
        cobblestone stop on a rainy night"
source_hash: f2cae41110f376c1
---

## 安装

```bash
pip install "libreyolo[gen]"
```

## 预测

镜像在首次使用时下载一次。LibreYOLO 用 diffusers 的 `Flux2KleinPipeline` 加载 OVB，
不添加任何网络代码：你要写的是提示词，以及几行把颜色读回来的 NumPy 代码。

<code-tabs name="predict" />

`source` 接受路径、URL、`PIL.Image` 或 BGR NumPy 数组。输出保持照片的宽高比，上限
为一百万像素；读取之前先把它缩放回照片尺寸（掩码用最近邻插值，这样颜色保持精确）。
不传 `source` 时，它就是一个普通的文生图模型。可调参数：`steps`（28）和
`guidance`（3.5），即论文里的设置。`offload=True` 会把闲置部分移出 GPU。

这个模型在每个任务的多种表述上训练过。库里的
[Open Vision Banana 指南](https://github.com/LibreYOLO/libreyolo/blob/v1.8.0/docs/open_vision_banana.md)
收录了 FunAILab 的快速上手提示词和读出方法，涵盖语义分割、指代分割和实例分割、深度、
表面法线和图像编辑，还有图例集成（用打乱的颜色跑几遍，再逐像素投票）。

预期效果：每个答案都是一次 28 步的扩散过程，所以很慢。当一个模型覆盖多个任务比速度
更重要时再用它。缩放后掩码的边缘是软的，并略有偏移；深度是相对逆深度，尺度和偏移
未知，法线是平滑的，但不是度量意义上的。

## 变体

| 尺寸 | 别名 | 镜像 | 文本编码器 | GPU 显存（库给出的数字） |
| --- | --- | --- | --- | --- |
| `4b` | `open-vision-banana` | `LibreYOLO/LibreOpenVisionBanana4b` | Qwen3 4B | 约 16 GB |
| `9b` | `open-vision-banana-9b` | `LibreYOLO/LibreOpenVisionBanana9b` | Qwen3 8B | 约 34 GB |

两个镜像存放的都是论文里的非零样本检查点（checkpoint），用基准测试的训练划分训练。
其他以 diffusers 流水线形式保存的 OVB 检查点（即零样本的那些）可以用
`LibreGen("open-vision-banana", checkpoint_dir="path/to/folder")` 加载。

## 导出

<export-matrix />

不支持训练、导出和 `val()`。

## 许可证

<provenance-box>

Open Vision Banana 出自 FunAILab
（[OpenVisionBanana/OpenVisionBanana](https://github.com/OpenVisionBanana/OpenVisionBanana)）。
权重采用 Black Forest Labs 的 **FLUX Non-Commercial License**：不得用于商业或生产
用途，第 2(e) 条要求在分享生成图像之前进行内容过滤或人工审核。FunAILab 的训练代码
采用 Apache-2.0 许可，但 LibreYOLO 没有移植其中任何部分。

LibreYOLO 的镜像是完整的 diffusers 流水线，不需要申请访问：FunAILab 的 transformer
从 float32 转换为 bfloat16，即它运行时的精度，BFL 的 FLUX.2 [klein] Base 流水线其余
部分逐字节一致。在相同 seed 下，输出与 FunAILab 自己的快速上手示例逐位一致。

</provenance-box>

## 引用

<citation-block />
