---
title: CRM
families:
  - crm
seo_title: 在 Python 里用 CRM：一张图片生成带颜色的 3D 网格，可选 UV 纹理
description: >-
  在 LibreYOLO 里运行清华大学的 CRM：输入一张物体图片，输出带颜色的三角网格 .obj，可选烘焙纹理。纯 PyTorch；带
  OpenRAIL++-M 使用限制。
lead: >-
  CRM（Convolutional Reconstruction Model）是清华大学的图像转 3D
  模型。两个多视图扩散模型生成物体的六个视图及其坐标图，一个卷积 UNet 把它们转成三平面（triplane），再由 FlexiCubes
  提取带颜色的网格。LibreYOLO 通过 LibreGen 工厂用纯 PyTorch 运行它。
keywords:
  - CRM
  - Convolutional Reconstruction Model
  - 清华 CRM
  - 图像转 3D
  - 图片生成 3D 模型
  - 带纹理网格 mesh
  - FlexiCubes
  - 多视图扩散
  - OBJ
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("crm")             # 权重在首次使用时下载
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")           # 顶点颜色写成 `v x y z r g b`
        m = result.meshes
        print(m.vertices[0].shape, m.extras["vertex_colors"][0].shape)
    - label: CLI
      language: bash
      code: >
        # 把物体以 .obj 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=crm seed=0

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=crm texture=true
    - label: 纹理
      language: python
      code: |
        # pip install xatlas   （MIT 许可，用于 UV 展开）
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("crm")
        model(SAMPLE_IMAGE, texture=True).save("object.obj")
        # 写出 object.obj、object.mtl 和 object_texture.png
source_hash: 9bc27f19244a7545
---

## 安装

```bash
pip install "libreyolo[gen]"
```

`texture=True` 还需要采用 MIT 许可的 `xatlas` 包（`pip install xatlas`）来做 UV 展开。

## 预测

没有任何内容需要申请访问，也不需要登录。权重在首次使用时从 [LibreYOLO/LibreCRM](https://huggingface.co/LibreYOLO/LibreCRM) 的固定版本下载。

<code-tabs name="predict" />

两个多视图扩散模型先生成物体的六个正交视图，再生成它们的规范坐标图；一个卷积 UNet 把它们转成三平面（triplane），FlexiCubes 再提取一个按三平面上色的网格。上游需要 xformers、nvdiffrast（英伟达，非商用）和一个 GPL 网格库；LibreYOLO 用纯 PyTorch 运行同样的权重，支持英伟达 GPU、Apple Silicon 和 CPU，并像上游一样在 GPU 上使用 fp16。

输入图像会在灰色背景上补成正方形，不做裁剪，所以照片里物体占满画面时效果最好。同一个种子在每种设备上给出相同的初始噪声；在 CUDA 上重复调用会得到相同的视图和三角形，顶点可能有约 1e-7 的偏移。库在 RTX 3090 上测得每个物体约 12 秒、12 GB。

`result.meshes` 包含 `vertices` (V, 3)，z 轴朝上，范围 `[-0.5, 0.5]^3`；`faces` (F, 3)；以及 `extras["vertex_colors"]` (V, 3)，RGB 值在 `[0, 1]` 内。`texture=True` 会像上游一样烘焙一张 1024 x 1024 的 UV 纹理。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `steps` | `50` | 两个扩散阶段的采样步数，1 到 499。 |
| `guidance` | `5.0` | 两个阶段的无分类器引导（classifier-free guidance）。 |
| `texture` | `False` | 烘焙 UV 纹理（在 .obj 旁边写出 .mtl 和 .png）。 |

## 变体

一个检查点（checkpoint），41 亿参数：两个 20 亿参数的多视图扩散模型（各含 UNet、OpenCLIP ViT-H/14 和 SD VAE），以及一个 1.2 亿参数的重建 UNet。约 12.8 GB，BiRefNet 另需 0.4 GB。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`（见库里的 ADR 0031）。

## 许可证

<provenance-box>

CRM 出自清华大学（TSAIL）（[thu-ml/CRM](https://github.com/thu-ml/CRM)）。代码采用 MIT 许可，模型卡也声明权重采用 MIT 许可。

**这些权重带有使用限制。** CRM 的两个扩散模型源自 ImageDream、MVDream 和 Stable Diffusion 2.1，所以 LibreYOLO 把它们视为适用这一谱系的 CreativeML OpenRAIL++-M License。允许商用，但其基于用途的限制（不得用于非法、有害、歧视性或欺骗性用途等）对你有约束力，而且你把模型或衍生作品分享给任何人时都必须把这些限制传递下去。

`Zhengyi/CRM` 权重以 LibreYOLO/LibreCRM 原样再分发，其模型卡带有 OpenRAIL++-M 条款。FlexiCubes 来自英伟达采用 Apache-2.0 许可的发布；上游调用的 nvdiffrast 和 GPL-3.0 网格工具都没有使用。

</provenance-box>

## 引用

<citation-block />
