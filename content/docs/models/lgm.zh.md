---
title: LGM
families:
  - lgm
seo_title: 在 Python 里用 LGM：经由四个生成视图把一张图片变成 3D 高斯
description: >-
  在 LibreYOLO 里运行 LGM：ImageDream 从一张图片采样出物体的四个视图，LGM 再把它们转成 3D 高斯。模型采用 MIT
  许可；ImageDream 带 OpenRAIL 使用限制。
lead: >-
  LGM（Large Multi-View Gaussian Model）分两个阶段把一张物体图片转成 3D 高斯泼溅（Gaussian
  Splatting）用的 3D 高斯：ImageDream 先采样物体周围的四个视图，LGM 再读取这些视图并返回高斯。LibreYOLO 通过
  LibreGen 工厂用纯 PyTorch 运行这两个阶段。
keywords:
  - LGM
  - Large Multi-View Gaussian Model
  - ImageDream
  - 图像转 3D
  - 3D 高斯泼溅 gaussian splatting
  - 高斯 splats
  - 多视图扩散
  - PLY
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("lgm")             # 权重在首次使用时下载
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # 标准 3DGS 布局
        print(len(result.gaussians))        # 不透明度高于 0.005 的那些
    - label: CLI
      language: bash
      code: >
        # 把物体以 .ply 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=lgm seed=0
    - label: 设置
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("lgm")
        result = model(SAMPLE_IMAGE, elevation=-10)     # 从上方拍摄的图片
        result = model(SAMPLE_IMAGE, steps=20, guidance=1.0)   # 不用引导
source_hash: 90133179ce28932d
---

## 安装

```bash
pip install "libreyolo[gen]"
```

这个 extra 会带上 `diffusers` 和 `transformers`，用来加载 ImageDream 的 VAE、调度器和 CLIP 编码器。

## 预测

没有任何内容需要申请访问，也不需要登录。权重在首次使用时按固定版本下载。

<code-tabs name="predict" />

ImageDream 是一个从 Stable Diffusion 2.1 微调而来的多视图扩散模型，它从 256 px 的输入采样出物体周围的四个视图。LGM 是一个带跨视图注意力的非对称 UNet，它读取这四个视图及其相机射线，为每个视图的每个输出像素返回一个高斯：4 x 128 x 128 = 65,536 个高斯，其中不透明度高于 0.005 的会被保留，与上游保存的方式一致。

上游在 CUDA 上需要 xformers，并用一个非商用的光栅化器渲染。LibreYOLO 通过 PyTorch 的 SDPA 运行注意力，并且不做渲染，所以模型可以在英伟达 GPU、Apple Silicon 和 CPU 上运行，GPU 上用 fp16，CPU 上用 fp32。在 CUDA 上，一个种子能复现上游的物体；其他设备在同一个种子下会抽到不同的噪声。

输出是 `result.gaussians`，与 [Speridlabs Idol](/docs/models/speridlabs-idol) 的载荷相同，可保存为 `.ply` 或 `.splat`。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `steps` | `30` | ImageDream 的 DDIM 采样步数。 |
| `guidance` | `5.0` | 无分类器引导（classifier-free guidance）；设为 1 即关闭。 |
| `elevation` | `0` | 拍摄图片时的相机高度，单位为度；负值表示在物体上方。 |

## 变体

一个尺寸，`0.4b`：4.2 亿参数的 LGM 网络（约 0.8 GB），加上 13 亿参数的 ImageDream 多视图阶段及其两个 CLIP 编码器（约 4.0 GB）。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`（见库里的 ADR 0031）。

## 许可证

<provenance-box>

LGM 出自 3D Topia 的 Jiaxiang Tang 等人（[3DTopia/LGM](https://github.com/3DTopia/LGM)）。两个阶段的许可证不同：

- LGM 高斯模型及其代码采用 MIT 许可。
- **ImageDream 带有使用限制。** 上游声明它采用 OpenRAIL 许可，LibreYOLO 按其 Stable Diffusion 2.1 谱系适用 CreativeML OpenRAIL++-M License：允许商用，但其基于用途的限制（不得用于非法、有害、歧视性或欺骗性用途等）对你有约束力，并且必须随模型或衍生作品一起传递下去。

权重按固定版本原样再分发：`ashawkey/LGM` 与 BiRefNet（MIT）一起作为 LibreYOLO/LibreLGM0.4b，`ashawkey/imagedream-ipmv-diffusers` 作为 [LibreYOLO/LibreImageDream](https://huggingface.co/LibreYOLO/LibreImageDream)。两者都不需要申请访问。上游的渲染器需要 Inria 的非商用光栅化器，因此没有包含在内。

</provenance-box>

## 引用

<citation-block />
