---
title: TripoSplat
families:
  - triposplat
seo_title: 在 Python 里用 TripoSplat：一张图片生成 3D 高斯 splats
description: >-
  在 LibreYOLO 里运行 TripoSplat：输入一张物体图片，把整个物体输出为最多 262,144 个 3D 高斯，保存为 .ply 或
  .splat。采用 MIT 许可；DINOv3 编码器适用 Meta 的许可证。
lead: >-
  TripoSplat 是 TripoAI（VAST）的图像转 3D 模型。它接收一张物体图片，把整个物体以 3D 高斯泼溅（Gaussian
  Splatting）的形式返回。Speridlabs Idol 就是基于它构建的，LibreYOLO 用同样的 LibreGen 调用和输出来运行它。
keywords:
  - TripoSplat
  - TripoAI
  - VAST
  - 图像转 3D
  - 3D 高斯泼溅 gaussian splatting
  - 高斯 splats
  - 3DGS
  - PLY
  - splat
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposplat")      # 权重在首次使用时下载
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # 标准 3DGS 布局
        result.save("object.splat")         # 紧凑的 Web 格式
        print(len(result.gaussians))
    - label: CLI
      language: bash
      code: >
        # 把物体以 .ply 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=triposplat seed=0
    - label: 设置
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposplat")
        result = model(SAMPLE_IMAGE, steps=20, gaussians=131072)
source_hash: 0e7cb2342a17effc
---

## 安装

```bash
pip install "libreyolo[gen]"
```

这个 extra 会带上 `diffusers`（FLUX.2 VAE）和 `transformers`（DINOv3 编码器）。

## 预测

没有任何内容需要申请访问，也不需要登录。权重在首次使用时按固定版本下载。

<code-tabs name="predict" />

一个 6.6 亿参数的 rectified-flow transformer 以 1024 px 输入的 DINOv3 和 FLUX.2 VAE token 为条件，对物体潜变量和输入相机去噪。一个八叉树解码器放置锚点，一个高斯解码器把每个锚点转成 32 个高斯，总数从 32,768 到 262,144。它在 CUDA 和 Apple MPS 上用 fp16 运行，在 CPU 上用 fp32 运行，没有编译路径。

输出与 [Speridlabs Idol](/docs/models/speridlabs-idol) 相同：`result.gaussians`，包含 `xyz`、`scales`、`rotations`、`opacities`、`sh`、`colors`、推断出的 `camera`、`seed` 和 `params`。物体 z 轴朝上，居中于 `[-0.5, 0.5]^3`。`.ply` 是转成 y 轴朝上的标准 3DGS 布局；`.splat` 是紧凑的 Web 格式。

种子像上游一样驱动模型所在设备上的生成器，所以同一个种子在不同设备上会给出不同的物体。与上游不同，八叉树采样也从带种子的生成器中抽取，所以一个种子能复现整个物体。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `steps` | `20` | 采样步数；上游建议 10 到 20。 |
| `guidance` | `3.0` | 无分类器引导（classifier-free guidance）；小于等于 1 即关闭。 |
| `shift` | `3.0` | flow 时间偏移。 |
| `gaussians` | `262144` | 输出中的高斯数量，从 32,768 到 262,144。 |
| `margin` | `1.2` | 裁剪边长与物体尺寸之比，与上游一致。 |

## 变体

一个尺寸，`0.6b`（6.6 亿参数），约 1.8 GB，另加 Idol 也在用的 DINOv3 ViT-H+ 编码器（1.7 GB）和 FLUX.2 VAE 编码器（0.2 GB）。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`（见库里的 ADR 0031）。

## 许可证

<provenance-box>

TripoSplat 出自 TripoAI（[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)）。代码和权重采用 MIT 许可；LibreYOLO 把去噪器、解码器和 BiRefNet 文件逐字节一致地镜像为 LibreYOLO/LibreTripoSplat0.6b。FLUX.2 VAE 来自 black-forest-labs/FLUX.2-klein-4B，采用 Apache-2.0 许可。

**DINOv3 图像编码器不是 MIT 许可。** 它属于 Meta，适用 DINOv3 License，这是一份带有贸易管制和最终用途限制（其中包括军事、武器、间谍和核用途）的自定义协议。LibreYOLO 把它以 bfloat16 连同该协议镜像在 LibreYOLO/LibreDINOv3vith16plus，所以没有任何内容需要申请访问。在基于这个模型发布产品之前，请先读一遍。

</provenance-box>

## 引用

<citation-block />
