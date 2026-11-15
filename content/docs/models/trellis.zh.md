---
title: TRELLIS
families:
  - trellis
seo_title: 在 Python 里用 TRELLIS：一张图片生成 3D 高斯，无需 CUDA 扩展
description: >-
  在 LibreYOLO 里运行微软的 TRELLIS：输入一张物体图片，输出稀疏体素结构上的 3D 高斯。纯 PyTorch，支持 CUDA、MPS 和
  CPU，采用 MIT 许可。
lead: >-
  TRELLIS 是微软开源的图像转 3D 模型。它先把物体构建成稀疏体素结构，再为每个被占据的体素生成 32 个 3D 高斯泼溅（Gaussian
  Splatting）用的高斯。上游需要只支持 CUDA 的扩展；LibreYOLO 通过 LibreGen 工厂用纯 PyTorch 运行同样的权重。
keywords:
  - TRELLIS
  - 微软 TRELLIS
  - 结构化 3D 潜变量 structured latents
  - 图像转 3D
  - 3D 高斯泼溅 gaussian splatting
  - 高斯 splats
  - 稀疏体素
  - PLY
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis")         # 权重在首次使用时下载
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # 标准 3DGS 布局
        result.save("object.splat")         # 紧凑的 Web 格式
        print(len(result.gaussians))        # 由物体决定，不是可调参数
    - label: CLI
      language: bash
      code: >
        # 把物体以 .ply 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=trellis seed=0
    - label: 设置
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis")
        result = model(SAMPLE_IMAGE, steps=12)          # 更快的采样
        result = model(SAMPLE_IMAGE, guidance=3.0, margin=1.3)
source_hash: 1b2b957c271c561e
---

## 安装

```bash
pip install "libreyolo[gen]"
```

这个 extra 会带上 `transformers`，用来加载 DINOv2 图像编码器。

## 预测

没有任何内容需要申请访问，也不需要登录。权重在首次使用时按固定版本下载。

<code-tabs name="predict" />

一个稠密 flow transformer 在 16^3 上对物体的占据情况去噪，一个 3D CNN 把它解码为 64^3 网格中的活跃体素，一个稀疏 flow transformer 在这些体素上对结构化潜变量去噪，再由一个稀疏 transformer 把每个体素转成 32 个高斯。两个 flow 阶段都读取 518 px 输入的 DINOv2 token。因此高斯的数量由物体决定（通常为 200,000 到 600,000 个）。

上游的稀疏卷积（spconv）和注意力（xformers 或 flash-attn）只能在 CUDA 上运行。LibreYOLO 用 PyTorch 运算替换它们并保留参数名，所以发布的权重在英伟达 GPU、Apple Silicon 和 CPU 上都能原样加载。噪声像上游一样在 CPU 上抽取，所以同一个种子在每种设备上都给出同一个物体。

输出是 `result.gaussians`，与 [Speridlabs Idol](/docs/models/speridlabs-idol) 的载荷相同：`xyz`、`scales`、`rotations`、`opacities`、`sh`、`colors`、`seed` 和 `params`，z 轴朝上，范围 `[-0.5, 0.5]^3`。背景移除由 LibreYOLO 移植的 BiRefNet 完成。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `steps` | `25` | 采样步数，两个 flow 阶段共用。 |
| `guidance` | `5.0` | 无分类器引导（classifier-free guidance），两个 flow 阶段共用。 |
| `shift` | `3.0` | flow 时间偏移。 |
| `margin` | `1.2` | 裁剪边长与物体尺寸之比，与上游一致。 |

## 变体

一个尺寸，`l`（TRELLIS-image-large）：4 个网络共 13 亿参数，约 2.7 GB，另加 DINOv2 ViT-L/14 编码器的 1.2 GB。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`（见库里的 ADR 0031）。

## 许可证

<provenance-box>

TRELLIS 出自微软研究院（[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)）。代码和 `microsoft/TRELLIS-image-large` 权重采用 MIT 许可，所以保留声明即可商用。LibreYOLO 把这些权重与 BiRefNet 权重（MIT）一起原样再分发为 LibreYOLO/LibreTRELLISl。DINOv2 编码器 `facebook/dinov2-with-registers-large` 采用 Apache-2.0 许可，从 Meta 下载。上游的高斯容器采用 Inria 的非商用 Gaussian Splatting 许可，没有使用。

</provenance-box>

## 引用

<citation-block />
