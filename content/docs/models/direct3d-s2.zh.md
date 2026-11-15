---
title: Direct3D-S2
families:
  - direct3d_s2
seo_title: 在 Python 里用 Direct3D-S2：一张图片生成 1024^3 精细 3D 网格
description: >-
  在 LibreYOLO 里运行 DreamTech 的 Direct3D-S2：输入一张物体图片，从最高 1024^3 的稀疏有符号距离场输出精细网格。纯
  PyTorch，采用 MIT 许可。
lead: >-
  Direct3D-S2 是 DreamTech 的图像转 3D 模型。它在稀疏体素上生成有符号距离场，先在 512^3，再在
  1024^3，最后把物体作为网格返回。上游需要只支持 CUDA 的扩展；LibreYOLO 通过 LibreGen 工厂用纯 PyTorch
  运行同样的权重。
keywords:
  - Direct3D-S2
  - DreamTech
  - 图像转 3D
  - 图片生成 3D 模型
  - 3D 网格生成 mesh
  - 有符号距离场 SDF
  - 空间稀疏注意力
  - OBJ
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("direct3d-s2")     # 权重在首次使用时下载
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape)
    - label: CLI
      language: bash
      code: >
        # 把物体以 .obj 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=direct3d-s2 seed=0
    - label: 设置
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("direct3d-s2")
        result = model(SAMPLE_IMAGE, resolution=512)     # 第一遍之后就停止
        result = model(SAMPLE_IMAGE, steps=(25, 15, 8))  # 每个阶段用更少的步数
source_hash: e90a338361a40563
---

## 安装

```bash
pip install "libreyolo[gen]"
```

这个 extra 会带上 `diffusers`、`transformers`（DINOv2 编码器）和 scikit-image（marching cubes）。

## 预测

没有任何内容需要申请访问，也不需要登录。权重在首次使用时按固定版本下载。

<code-tabs name="predict" />

一个稠密 flow transformer 找出被占据的 64^3 体素，一个带空间稀疏注意力的稀疏 flow transformer 在这些体素上对潜变量去噪，一个稀疏解码器再把它转成 512^3 的有符号距离场。第二遍在 1024^3 上重复这一过程。网格是该场的零值面，由 marching cubes 提取。上游只能在 CUDA 上运行（torchsparse、flash-attn、Triton 内核、一个 CUDA 体素化器）；LibreYOLO 用 PyTorch 运算替换它们并保留参数名，所以发布的检查点（checkpoint）在 CUDA、Apple MPS 和 CPU 上都能原样加载。

同一个种子给出同一个物体；在 CUDA 上，初始噪声与上游在 `torch.manual_seed(seed)` 下的一致。`result.meshes` 包含 `vertices` (V, 3)，z 轴朝上，范围 `[-0.5, 0.5]^3`，以及 `faces` (F, 3)。上游可选的重新网格化和补洞没有包含在内（它们依赖 GPL 工具），所以网格就是 marching cubes 返回的样子。

纯 PyTorch 下它很慢：库在 RTX A6000 上测得一个 1024^3 的物体约需 18 分钟和 23 GB 显存，而上游的 CUDA 内核只要 2.5 分钟。`resolution=512` 会在第一遍之后停止。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `resolution` | `1024` | 1024（上游默认）或 512。 |
| `steps` | `(50, 30, 15)` | 稠密、512 和 1024 三个阶段的步数；传一个整数则三个阶段都用它。 |
| `guidance` | `7.0` | 三个阶段的无分类器引导（classifier-free guidance）。 |

## 变体

一个检查点，`v11`（Direct3D-S2 v1.1）：8 个网络共 20 亿参数，约 4.2 GB，另加 DINOv2 ViT-L/14 的 1.2 GB 和 BiRefNet 的 0.4 GB。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`（见库里的 ADR 0031）。

## 许可证

<provenance-box>

Direct3D-S2 出自 DreamTech（[DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)）。代码和 `wushuang98/Direct3D-S2` v1.1 权重采用 MIT 许可，所以保留声明即可商用。LibreYOLO 以 LibreYOLO/LibreDirect3DS2v11 原样再分发这些权重。BiRefNet（MIT）来自 LibreYOLO/LibreTRELLISl，DINOv2 编码器 `facebook/dinov2-with-registers-large`（Apache-2.0）来自 Meta。上游采用 GPL 许可的 pymeshfix 和 python-igraph 都没有使用。

</provenance-box>

## 引用

<citation-block />
