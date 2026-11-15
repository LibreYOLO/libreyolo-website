---
title: TripoSR
families: [triposr]
seo_title: "TripoSR in Python: one image to a 3D mesh in about a second"
description: "Run TripoSR in LibreYOLO: one picture of an object in, a vertex-coloured triangle mesh out as .obj. Feed-forward, no sampling, MIT code and weights."
lead: "TripoSR is a feed-forward image-to-3D model by Tripo AI and Stability AI. It turns one picture of an object into a triplane NeRF in a single pass and returns the object as a vertex-coloured triangle mesh. LibreYOLO runs it through the LibreGen factory in plain PyTorch."
keywords: [TripoSR, Tripo AI, Stability AI, image to 3D, single image to 3D, mesh generation, triplane NeRF, OBJ, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposr")        # weights download on first use
        result = model(SAMPLE_IMAGE)

        result.save("object.obj")          # vertex colours as `v x y z r g b`
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape)   # (V, 3), (F, 3)
    - label: CLI
      language: bash
      code: |
        # Saves the object as .obj under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=triposr
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposr")
        result = model(SAMPLE_IMAGE, resolution=384)    # finer marching-cubes grid
        result = model(SAMPLE_IMAGE, threshold=25.0)    # density at the surface
---

## Install

TripoSR runs on the LibreGen extra, which brings scikit-image for marching
cubes.

```bash
pip install "libreyolo[gen]"
```

## Predict

Weights download on first use from
[LibreYOLO/LibreTripoSR](https://huggingface.co/LibreYOLO/LibreTripoSR), with
nothing gated and no login.

<code-tabs name="predict" />

A DINO ViT-B/16 encodes the 512 px input, a transformer decodes a triplane
NeRF from it in one pass, and the mesh is the marching-cubes isosurface of the
NeRF's density. There is no sampling, so `seed` is accepted like every
LibreGen family but changes nothing. It takes about a second on a GPU and runs
in fp32 on NVIDIA GPUs, Apple silicon and CPU.

The source is anything predict takes: a path, URL, PIL image, NumPy array, a
folder or a list. An image with real transparency is used as the cutout;
anything else has its background removed first with LibreYOLO's BiRefNet port.

`result.meshes` holds one mesh per object: `vertices` (V, 3), z-up inside
`[-0.5, 0.5]^3`, `faces` (F, 3), and `extras["vertex_colors"]` (V, 3) RGB in
`[0, 1]`. The input view looks along -x, so the front of the object faces +x.
`save()` writes `.obj` with vertex colours.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `resolution` | `256` | Marching-cubes grid, from 32 to 1024. |
| `threshold` | `25.0` | Density at the surface. |
| `margin` | `1/0.85` | Crop side over object size (upstream's foreground ratio 0.85). |

Marching cubes runs through scikit-image instead of upstream's torchmcubes:
the same isosurface, triangulated differently.

## Variants

One checkpoint, 0.42B parameters, about 1.7 GB plus 0.4 GB for BiRefNet.

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError`; their names are reserved by ADR 0031 in
the library.

## Licensing

<provenance-box>

TripoSR is by Tripo AI and Stability AI. The code
([VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)) and
the `stabilityai/TripoSR` weights are MIT, so commercial use is fine with the
copyright notice kept. LibreYOLO redistributes the weights unchanged as
LibreYOLO/LibreTripoSR. It avoids upstream's torchmcubes (MPL-2.0) by using
scikit-image (BSD-3-Clause); the BiRefNet background remover weights are MIT.

</provenance-box>

## Citation

<citation-block />
