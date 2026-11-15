---
title: TRELLIS.2
families:
  - trellis2
seo_title: 在 Python 里用 TRELLIS.2：一张图片生成最高 1536^3 的 3D 网格
description: >-
  在 LibreYOLO 里运行微软的 TRELLIS.2 4B：输入一张物体图片，输出形状的三角网格。纯 PyTorch，支持 CUDA、MPS 和
  CPU。代码采用 MIT 许可；DINOv3 编码器适用 Meta 的许可证。
lead: >-
  TRELLIS.2 是微软开源的 40 亿参数图像转 3D 模型。它把物体的形状作为网格返回，网格分辨率最高 1536^3。LibreYOLO 通过
  LibreGen 工厂用纯 PyTorch 运行同样的权重，只输出几何。
keywords:
  - TRELLIS.2
  - TRELLIS 2
  - 微软 TRELLIS
  - 图像转 3D
  - 3D 网格生成 mesh
  - 结构化潜变量 structured latents
  - OBJ
  - DINOv3
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis2")        # 也可以写 "trellis.2"
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape, m.seed)
    - label: CLI
      language: bash
      code: >
        # 把物体以 .obj 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=trellis2 seed=0
    - label: 设置
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis2")
        result = model(SAMPLE_IMAGE, resolution=512)    # 只有一个形状阶段，更快
        result = model(SAMPLE_IMAGE, resolution=1536)   # 最精细的网格
source_hash: 9f55834ac4ceb0cd
---

## 安装

```bash
pip install "libreyolo[gen]"
```

这个 extra 会带上 `transformers`，用来加载 DINOv3 图像编码器。

## 预测

没有任何内容需要申请访问，也不需要登录。权重在首次使用时按固定版本下载。

<code-tabs name="predict" />

结构阶段确定被占据的体素，形状阶段在这些体素上填充潜变量（先在 512，再在 1024 或 1536 上重复一次），一个稀疏解码器再把它转成对偶网格（dual grid）上的网格。两个 flow 阶段都读取 DINOv3 ViT-L/16 的 token。上游需要只支持 CUDA 的扩展（FlexGEMM、flash-attn、o-voxel）；LibreYOLO 用纯 PyTorch 运行同样的权重，支持英伟达 GPU、Apple Silicon 和 CPU。噪声在 CPU 上抽取，所以同一个种子在每种设备上给出同一个物体，差别只在内核的舍入误差。

`result.meshes` 存放网格：`vertices` (V, 3)，z 轴朝上，范围 `[-0.5, 0.5]^3`，以及 `faces` (F, 3)。它只有几何，而且是原始的对偶网格，没有经过上游的 CuMesh 清理：面的绕序不一致，有些顶点没有被使用。上游的纹理阶段用 nvdiffrast（英伟达的非商用库）烘焙，不属于 LibreYOLO。

库在 A100 上测得一个 1024 的物体需要 19.5 秒、显存峰值 14.3 GB，512 时需要 5.5 秒、11.6 GB。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `steps` | `12` | 采样步数。 |
| `guidance` | `7.5` | 无分类器引导（classifier-free guidance）。 |
| `resolution` | `1024` | 输出网格分辨率：512、1024 或 1536。 |
| `margin` | `1.0` | 裁剪边长与物体尺寸之比。 |
| `max_tokens` | `49152` | 级联的 token 预算；为了放得下，它可能会降低分辨率，实际使用的分辨率记录在 `params` 里。 |

## 变体

一个尺寸，`4b`：5 个网络共 40 亿参数，约 8.7 GB，另加 DINOv3 编码器的 1.2 GB 和来自 TRELLIS 镜像的 1.3 GB。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`（见库里的 ADR 0031）。

## 许可证

<provenance-box>

TRELLIS.2 出自微软研究院（[microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)）。代码和 `microsoft/TRELLIS.2-4B` 权重采用 MIT 许可。

**图像编码器决定了整条流水线的条款。** 它是 Meta 的 `facebook/dinov3-vitl16-pretrain-lvd1689m`，适用 DINOv3 License。这是一份未经 OSI 批准的自定义协议，会向下传递贸易管制和最终用途限制（其中包括军事、武器、间谍和核用途）。在基于这个模型发布产品之前，请先读一遍。

权重按固定版本原样再分发，都不需要申请访问：几何网络作为 LibreYOLO/LibreTRELLIS2-4b，TRELLIS 的结构解码器和 BiRefNet（MIT）来自 LibreYOLO/LibreTRELLISl，DINOv3 编码器连同该协议作为 LibreYOLO/LibreDINOv3vitl16。nvdiffrast、nvdiffrec 和 RMBG-2.0（都是非商用）都没有使用。

</provenance-box>

## 引用

<citation-block />
