---
title: TripoSR
families:
  - triposr
seo_title: 在 Python 里用 TripoSR：约一秒把一张图片变成 3D 网格
description: 在 LibreYOLO 里运行 TripoSR：输入一张物体图片，输出带顶点颜色的三角网格 .obj。前馈式，无需采样，代码和权重都采用 MIT 许可。
lead: >-
  TripoSR 是 Tripo AI 和 Stability AI 推出的前馈式图像转 3D 模型。它一次前向就把一张物体图片转成三平面
  NeRF（triplane NeRF），并把物体作为带顶点颜色的三角网格返回。LibreYOLO 通过 LibreGen 工厂用纯 PyTorch 运行它。
keywords:
  - TripoSR
  - Tripo AI
  - Stability AI
  - 图像转 3D
  - 单张图片生成 3D 模型
  - 3D 网格生成 mesh
  - triplane NeRF
  - OBJ
  - LibreGen
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposr")        # 权重在首次使用时下载
        result = model(SAMPLE_IMAGE)

        result.save("object.obj")          # 顶点颜色写成 `v x y z r g b`
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape)   # (V, 3), (F, 3)
    - label: CLI
      language: bash
      code: >
        # 把物体以 .obj 保存到 runs/generate/predict 下

        libreyolo generate
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg
        model=triposr
    - label: 设置
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposr")
        result = model(SAMPLE_IMAGE, resolution=384)    # 更精细的 marching cubes 网格
        result = model(SAMPLE_IMAGE, threshold=25.0)    # 表面处的密度
source_hash: 6936728dac15269e
---

## 安装

TripoSR 使用 LibreGen extra，它会带上用于 marching cubes 的 scikit-image。

```bash
pip install "libreyolo[gen]"
```

## 预测

权重在首次使用时从 [LibreYOLO/LibreTripoSR](https://huggingface.co/LibreYOLO/LibreTripoSR) 下载，没有任何内容需要申请访问，也不需要登录。

<code-tabs name="predict" />

一个 DINO ViT-B/16 对 512 px 的输入编码，一个 transformer 一次前向从中解码出三平面 NeRF，网格就是 NeRF 密度的 marching cubes 等值面。这里没有采样，所以 `seed` 像每个 LibreGen 家族一样可以传入，但不会改变任何东西。它在 GPU 上大约需要一秒，在英伟达 GPU、Apple Silicon 和 CPU 上用 fp32 运行。

输入源可以是 predict 接受的任何东西：路径、URL、PIL 图像、NumPy 数组、文件夹或列表。真正带透明通道的图像会直接作为抠图使用；其他图像会先用 LibreYOLO 移植的 BiRefNet 移除背景。

`result.meshes` 为每个物体存放一个网格：`vertices` (V, 3)，z 轴朝上，位于 `[-0.5, 0.5]^3` 内；`faces` (F, 3)；以及 `extras["vertex_colors"]` (V, 3)，RGB 值在 `[0, 1]` 内。输入视角沿 -x 方向看，所以物体的正面朝向 +x。`save()` 写出带顶点颜色的 `.obj`。

### 设置

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `resolution` | `256` | marching cubes 网格分辨率，从 32 到 1024。 |
| `threshold` | `25.0` | 表面处的密度。 |
| `margin` | `1/0.85` | 裁剪边长与物体尺寸之比（上游的前景比例 0.85）。 |

marching cubes 通过 scikit-image 运行，而不是上游的 torchmcubes：等值面相同，只是三角化方式不同。

## 变体

一个检查点（checkpoint），4.2 亿参数，约 1.7 GB，BiRefNet 另需 0.4 GB。

## 导出

<export-matrix />

LibreGen 的 3D 家族没有实现导出。`export()`、`train()` 和 `val()` 会抛出 `NotImplementedError`；这些名字由库里的 ADR 0031 保留。

## 许可证

<provenance-box>

TripoSR 出自 Tripo AI 和 Stability AI。代码（[VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)）和 `stabilityai/TripoSR` 权重采用 MIT 许可，所以保留版权声明即可商用。LibreYOLO 把这些权重原样再分发为 LibreYOLO/LibreTripoSR。它用 scikit-image（BSD-3-Clause）替代了上游的 torchmcubes（MPL-2.0）；BiRefNet 背景移除权重采用 MIT 许可。

</provenance-box>

## 引用

<citation-block />
