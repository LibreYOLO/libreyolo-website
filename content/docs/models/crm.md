---
title: CRM
families: [crm]
seo_title: "CRM in Python: one image to a coloured 3D mesh, optional UV texture"
description: "Run Tsinghua's CRM in LibreYOLO: one picture of an object in, a coloured triangle mesh out as .obj, with an optional baked texture. Plain PyTorch; OpenRAIL++-M use restrictions."
lead: "CRM (Convolutional Reconstruction Model) is Tsinghua University's image-to-3D model. Two multi-view diffusion models generate six views of the object and their coordinate maps, a convolutional UNet turns them into a triplane, and FlexiCubes extracts a coloured mesh. LibreYOLO runs it in plain PyTorch through the LibreGen factory."
keywords: [CRM, Convolutional Reconstruction Model, Tsinghua, image to 3D, textured mesh, FlexiCubes, multi-view diffusion, OBJ, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("crm")             # weights download on first use
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")           # vertex colours as `v x y z r g b`
        m = result.meshes
        print(m.vertices[0].shape, m.extras["vertex_colors"][0].shape)
    - label: CLI
      language: bash
      code: |
        # Saves the object as .obj under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=crm seed=0
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=crm texture=true
    - label: Texture
      language: python
      code: |
        # pip install xatlas   (MIT, for the UV unwrap)
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("crm")
        model(SAMPLE_IMAGE, texture=True).save("object.obj")
        # writes object.obj, object.mtl and object_texture.png
---

## Install

```bash
pip install "libreyolo[gen]"
```

`texture=True` also needs the MIT-licensed `xatlas` package
(`pip install xatlas`) for the UV unwrap.

## Predict

Nothing is gated and no login is needed. Weights download on first use from
[LibreYOLO/LibreCRM](https://huggingface.co/LibreYOLO/LibreCRM) at a pinned
revision.

<code-tabs name="predict" />

Two multi-view diffusion models generate six orthographic views of the object
and then their canonical coordinate maps; a convolutional UNet turns them into
a triplane, and FlexiCubes extracts a mesh coloured by the triplane. Upstream
needs xformers, nvdiffrast (NVIDIA, non-commercial) and a GPL mesh library;
LibreYOLO runs the same weights in plain PyTorch on NVIDIA GPUs, Apple silicon
and CPU, in fp16 on the GPUs as upstream.

The input is padded to a square on grey, without a crop, so a photo works best
with the object filling the frame. A seed gives the same starting noise on
every device; on CUDA a repeat call gives the same views and triangles, with
vertices that can move by about 1e-7. The library measured about 12 s and 12 GB
per object on an RTX 3090.

`result.meshes` holds `vertices` (V, 3), z-up in `[-0.5, 0.5]^3`, `faces`
(F, 3) and `extras["vertex_colors"]` (V, 3) RGB in `[0, 1]`.
`texture=True` bakes a 1024 x 1024 UV texture, as upstream does.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `steps` | `50` | Sampling steps of both diffusion stages, 1 to 499. |
| `guidance` | `5.0` | Classifier-free guidance of both stages. |
| `texture` | `False` | Bake a UV texture (.mtl and .png beside the .obj). |

## Variants

One checkpoint, 4.1B parameters: two 2.0B multi-view diffusion models
(UNet, OpenCLIP ViT-H/14 and SD VAE each) and a 0.12B reconstruction UNet.
About 12.8 GB, plus 0.4 GB for BiRefNet.

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError` (ADR 0031 in the library).

## Licensing

<provenance-box>

CRM is by Tsinghua University (TSAIL)
([thu-ml/CRM](https://github.com/thu-ml/CRM)). The code is MIT and the model
card declares MIT for the weights.

**The weights carry use restrictions.** CRM's two diffusion models descend
from ImageDream, MVDream and Stable Diffusion 2.1, so LibreYOLO treats them as
under the CreativeML OpenRAIL++-M License of that lineage. Commercial use is
allowed, but its use-based restrictions (no illegal, harmful, discriminatory
or deceptive uses, among others) bind you and must be passed on to anyone you
share the model or derivatives with.

The `Zhengyi/CRM` weights are redistributed unchanged as LibreYOLO/LibreCRM,
whose card carries the OpenRAIL++-M terms. FlexiCubes comes from NVIDIA's
Apache-2.0 release; nvdiffrast and the GPL-3.0 mesh tools upstream calls are
not used.

</provenance-box>

## Citation

<citation-block />
