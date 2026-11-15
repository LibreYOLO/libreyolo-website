---
title: TRELLIS.2
families: [trellis2]
seo_title: "TRELLIS.2 in Python: one image to a 3D mesh up to 1536^3"
description: "Run Microsoft's TRELLIS.2 4B in LibreYOLO: one picture of an object in, the shape out as a triangle mesh. Plain PyTorch on CUDA, MPS and CPU. MIT code; DINOv3 encoder under Meta's licence."
lead: "TRELLIS.2 is Microsoft's open 4B image-to-3D model. It returns the object's shape as a mesh, on a grid of up to 1536^3. LibreYOLO runs the same weights in plain PyTorch through the LibreGen factory, geometry only."
keywords: [TRELLIS.2, TRELLIS 2, Microsoft, image to 3D, mesh generation, structured latents, OBJ, DINOv3, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis2")        # also "trellis.2"
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape, m.seed)
    - label: CLI
      language: bash
      code: |
        # Saves the object as .obj under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=trellis2 seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis2")
        result = model(SAMPLE_IMAGE, resolution=512)    # one shape stage, faster
        result = model(SAMPLE_IMAGE, resolution=1536)   # finest grid
---

## Install

```bash
pip install "libreyolo[gen]"
```

The extra brings `transformers`, which loads the DINOv3 image encoder.

## Predict

Nothing is gated and no login is needed. Weights download on first use at
pinned revisions.

<code-tabs name="predict" />

A structure stage places the occupied voxels, shape stages fill in a latent
on them (at 512, then again at 1024 or 1536), and a sparse decoder turns it
into a dual-grid mesh. Both flow stages read DINOv3 ViT-L/16 tokens. Upstream
needs CUDA-only extensions (FlexGEMM, flash-attn, o-voxel); LibreYOLO runs the
same weights in plain PyTorch on NVIDIA GPUs, Apple silicon and CPU. Noise is
drawn on the CPU, so a seed gives the same object on every device up to kernel
rounding.

`result.meshes` holds the mesh: `vertices` (V, 3), z-up in `[-0.5, 0.5]^3`,
and `faces` (F, 3). It is geometry only and the raw dual grid, before
upstream's CuMesh clean-up: faces are not consistently wound and some vertices
are unused. Upstream's texture stage bakes with nvdiffrast, a non-commercial
NVIDIA library, and is not part of LibreYOLO.

On an A100 the library measured 19.5 s and 14.3 GB peak GPU memory for one
object at 1024, and 5.5 s and 11.6 GB at 512.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `steps` | `12` | Sampling steps. |
| `guidance` | `7.5` | Classifier-free guidance. |
| `resolution` | `1024` | Output grid: 512, 1024 or 1536. |
| `margin` | `1.0` | Crop side over object size. |
| `max_tokens` | `49152` | Token budget for the cascade; it may lower the resolution to fit, and `params` records the resolution used. |

## Variants

One size, `4b`: 4B parameters in five networks, about 8.7 GB, plus 1.2 GB for
the DINOv3 encoder and 1.3 GB from the TRELLIS mirror.

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError` (ADR 0031 in the library).

## Licensing

<provenance-box>

TRELLIS.2 is by Microsoft Research
([microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)). The code and
the `microsoft/TRELLIS.2-4B` weights are MIT.

**The image encoder sets the terms for the pipeline.** It is Meta's
`facebook/dinov3-vitl16-pretrain-lvd1689m`, under the DINOv3 License, a
custom agreement that is not OSI-approved and passes through trade-control and
end-use restrictions (military, weapons, espionage and nuclear uses among
them). Read it before shipping a product on this model.

Weights are redistributed unchanged at pinned revisions and none is gated: the
geometry networks as LibreYOLO/LibreTRELLIS2-4b, TRELLIS's structure decoder
and BiRefNet (MIT) from LibreYOLO/LibreTRELLISl, and the DINOv3 encoder as
LibreYOLO/LibreDINOv3vitl16 with the agreement. nvdiffrast, nvdiffrec and
RMBG-2.0 (all non-commercial) are not used.

</provenance-box>

## Citation

<citation-block />
