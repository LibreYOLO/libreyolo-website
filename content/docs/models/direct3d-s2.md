---
title: Direct3D-S2
families: [direct3d_s2]
seo_title: "Direct3D-S2 in Python: one image to a detailed 3D mesh at 1024^3"
description: "Run DreamTech's Direct3D-S2 in LibreYOLO: one picture of an object in, a detailed mesh out from a sparse signed distance field at up to 1024^3. Plain PyTorch, MIT."
lead: "Direct3D-S2 is DreamTech's image-to-3D model. It generates a signed distance field on sparse voxels, first at 512^3 and then at 1024^3, and returns the object as a mesh. Upstream needs CUDA-only extensions; LibreYOLO runs the same weights in plain PyTorch through the LibreGen factory."
keywords: [Direct3D-S2, DreamTech, image to 3D, mesh generation, signed distance field, spatial sparse attention, OBJ, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("direct3d-s2")     # weights download on first use
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape)
    - label: CLI
      language: bash
      code: |
        # Saves the object as .obj under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=direct3d-s2 seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("direct3d-s2")
        result = model(SAMPLE_IMAGE, resolution=512)     # stops after the first pass
        result = model(SAMPLE_IMAGE, steps=(25, 15, 8))  # fewer steps per stage
---

## Install

```bash
pip install "libreyolo[gen]"
```

The extra brings `diffusers`, `transformers` (DINOv2 encoder) and
scikit-image (marching cubes).

## Predict

Nothing is gated and no login is needed. Weights download on first use at
pinned revisions.

<code-tabs name="predict" />

A dense flow transformer finds the occupied 64^3 voxels, a sparse flow
transformer with spatial sparse attention denoises a latent on them, and a
sparse decoder turns it into a signed distance field at 512^3. A second pass
repeats this at 1024^3. The mesh is the field's zero crossing, from marching
cubes. Upstream runs on CUDA only (torchsparse, flash-attn, Triton kernels, a
CUDA voxeliser); LibreYOLO replaces them with PyTorch operations and keeps the
parameter names, so the released checkpoints load unchanged on CUDA, Apple MPS
and CPU.

A seed gives the same object; on CUDA the starting noise equals upstream's
for `torch.manual_seed(seed)`. `result.meshes` holds `vertices` (V, 3), z-up
in `[-0.5, 0.5]^3`, and `faces` (F, 3). Upstream's optional remeshing and
hole filling are not included (they depend on GPL tools), so meshes come as
marching cubes returns them.

It is slow in plain PyTorch: on an RTX A6000 the library measured about 18
minutes and 23 GB of GPU memory for one object at 1024^3, where upstream's
CUDA kernels take 2.5 minutes. `resolution=512` stops after the first pass.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `resolution` | `1024` | 1024 (upstream's default) or 512. |
| `steps` | `(50, 30, 15)` | Steps of the dense, 512 and 1024 stages; one integer sets all three. |
| `guidance` | `7.0` | Classifier-free guidance of all three stages. |

## Variants

One checkpoint, `v11` (Direct3D-S2 v1.1): 2.0B parameters in eight networks,
about 4.2 GB, plus 1.2 GB for DINOv2 ViT-L/14 and 0.4 GB for BiRefNet.

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError` (ADR 0031 in the library).

## Licensing

<provenance-box>

Direct3D-S2 is by DreamTech
([DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)). Code
and the `wushuang98/Direct3D-S2` v1.1 weights are MIT, so commercial use is
fine with the notice kept. LibreYOLO redistributes the weights unchanged as
LibreYOLO/LibreDirect3DS2v11. BiRefNet (MIT) comes from LibreYOLO/LibreTRELLISl
and the DINOv2 encoder, `facebook/dinov2-with-registers-large` (Apache-2.0),
from Meta. Upstream's GPL pymeshfix and python-igraph are not used.

</provenance-box>

## Citation

<citation-block />
