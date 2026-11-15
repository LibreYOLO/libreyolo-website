---
title: TRELLIS
families: [trellis]
seo_title: "TRELLIS in Python: one image to 3D Gaussians, no CUDA extensions"
description: "Run Microsoft's TRELLIS in LibreYOLO: one picture of an object in, the object out as 3D Gaussians on a sparse voxel structure. Plain PyTorch on CUDA, MPS and CPU, MIT."
lead: "TRELLIS is Microsoft's open image-to-3D model. It builds the object as a sparse voxel structure first, then 32 Gaussians per occupied voxel. Upstream needs CUDA-only extensions; LibreYOLO runs the same weights in plain PyTorch through the LibreGen factory."
keywords: [TRELLIS, Microsoft, structured 3D latents, image to 3D, 3D Gaussian splatting, Gaussian splats, sparse voxels, PLY, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis")         # weights download on first use
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # standard 3DGS layout
        result.save("object.splat")         # compact web format
        print(len(result.gaussians))        # follows the object, not a knob
    - label: CLI
      language: bash
      code: |
        # Saves the object as .ply under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=trellis seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis")
        result = model(SAMPLE_IMAGE, steps=12)          # faster sampling
        result = model(SAMPLE_IMAGE, guidance=3.0, margin=1.3)
---

## Install

```bash
pip install "libreyolo[gen]"
```

The extra brings `transformers`, which loads the DINOv2 image encoder.

## Predict

Nothing is gated and no login is needed. Weights download on first use at
pinned revisions.

<code-tabs name="predict" />

A dense flow transformer denoises the object's occupancy at 16^3, a 3D CNN
decodes it to the active voxels of a 64^3 grid, a sparse flow transformer
denoises a structured latent on those voxels, and a sparse transformer turns
each voxel into 32 Gaussians. Both flow stages read DINOv2 tokens of the
518 px input. The number of Gaussians therefore follows the object (often
200,000 to 600,000).

Upstream's sparse convolutions (spconv) and attention (xformers or flash-attn)
run on CUDA only. LibreYOLO replaces them with PyTorch operations and keeps
the parameter names, so the released weights load unchanged on NVIDIA GPUs,
Apple silicon and CPU. Noise is drawn on the CPU as upstream does, so a seed
gives the same object on every device.

The output is `result.gaussians`, the same payload as
[Speridlabs Idol](/docs/models/speridlabs-idol): `xyz`, `scales`,
`rotations`, `opacities`, `sh`, `colors`, `seed` and `params`, z-up in
`[-0.5, 0.5]^3`. Background removal runs on LibreYOLO's BiRefNet port.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `steps` | `25` | Sampling steps, for both flow stages. |
| `guidance` | `5.0` | Classifier-free guidance, for both flow stages. |
| `shift` | `3.0` | Flow time shift. |
| `margin` | `1.2` | Crop side over object size, as upstream. |

## Variants

One size, `l` (TRELLIS-image-large): 1.3B parameters in four networks, about
2.7 GB, plus 1.2 GB for the DINOv2 ViT-L/14 encoder.

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError` (ADR 0031 in the library).

## Licensing

<provenance-box>

TRELLIS is by Microsoft Research
([microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)). Code and the
`microsoft/TRELLIS-image-large` weights are MIT, so commercial use is fine
with the notice kept. LibreYOLO redistributes the weights unchanged, with the
BiRefNet weights (MIT), as LibreYOLO/LibreTRELLISl. The DINOv2 encoder,
`facebook/dinov2-with-registers-large`, is Apache-2.0 and downloads from
Meta. Upstream's Gaussian holder, under Inria's non-commercial Gaussian
Splatting license, is not used.

</provenance-box>

## Citation

<citation-block />
