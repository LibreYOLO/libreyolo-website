---
title: TripoSplat
families: [triposplat]
seo_title: "TripoSplat in Python: one image to 3D Gaussian splats"
description: "Run TripoSplat in LibreYOLO: one picture of an object in, the whole object out as up to 262,144 3D Gaussians in .ply or .splat. MIT; DINOv3 encoder under Meta's licence."
lead: "TripoSplat is an image-to-3D model by TripoAI (VAST). It takes one picture of an object and returns the whole object as 3D Gaussian splats. It is the model Speridlabs Idol builds on, and LibreYOLO runs it through the same LibreGen call and output."
keywords: [TripoSplat, TripoAI, VAST, image to 3D, 3D Gaussian splatting, Gaussian splats, 3DGS, PLY, splat, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposplat")      # weights download on first use
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # standard 3DGS layout
        result.save("object.splat")         # compact web format
        print(len(result.gaussians))
    - label: CLI
      language: bash
      code: |
        # Saves the object as .ply under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=triposplat seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposplat")
        result = model(SAMPLE_IMAGE, steps=20, gaussians=131072)
---

## Install

```bash
pip install "libreyolo[gen]"
```

The extra brings `diffusers` (FLUX.2 VAE) and `transformers` (DINOv3 encoder).

## Predict

Nothing is gated and no login is needed. Weights download on first use at
pinned revisions.

<code-tabs name="predict" />

A 0.66B rectified-flow transformer denoises an object latent and the input
camera, conditioned on DINOv3 and FLUX.2 VAE tokens of the 1024 px input. An
octree decoder places anchors and a Gaussian decoder turns each anchor into 32
Gaussians, from 32,768 to 262,144 in all. It runs in fp16 on CUDA and Apple
MPS and in fp32 on CPU, with no compiled path.

The output is the same as [Speridlabs Idol](/docs/models/speridlabs-idol):
`result.gaussians` with `xyz`, `scales`, `rotations`, `opacities`, `sh`,
`colors`, the inferred `camera`, `seed` and `params`. The object is z-up and
centred in `[-0.5, 0.5]^3`. `.ply` is the standard 3DGS layout turned y-up;
`.splat` is the compact web format.

The seed drives a generator on the model's device, as upstream does, so one
seed gives different objects on different devices. Unlike upstream, the octree
sampling also draws from the seeded generator, so a seed reproduces the whole
object.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `steps` | `20` | Sampling steps; upstream recommends 10 to 20. |
| `guidance` | `3.0` | Classifier-free guidance; 1 or less turns it off. |
| `shift` | `3.0` | Flow time shift. |
| `gaussians` | `262144` | Gaussians in the output, from 32,768 to 262,144. |
| `margin` | `1.2` | Crop side over object size, as upstream. |

## Variants

One size, `0.6b` (0.66B parameters), about 1.8 GB plus the DINOv3 ViT-H+
encoder (1.7 GB) and the FLUX.2 VAE encoder (0.2 GB) that Idol also uses.

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError` (ADR 0031 in the library).

## Licensing

<provenance-box>

TripoSplat is by TripoAI
([VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)).
Code and weights are MIT; LibreYOLO mirrors the denoiser, decoder and BiRefNet
files byte-identical as LibreYOLO/LibreTripoSplat0.6b. The FLUX.2 VAE comes
from black-forest-labs/FLUX.2-klein-4B under Apache-2.0.

**The DINOv3 image encoder is not MIT.** It is Meta's, under the DINOv3
License, a custom agreement with trade-control and end-use restrictions
(military, weapons, espionage and nuclear uses among them). LibreYOLO mirrors
it in bfloat16 at LibreYOLO/LibreDINOv3vith16plus with the agreement, so
nothing is gated. Read it before shipping a product on this model.

</provenance-box>

## Citation

<citation-block />
