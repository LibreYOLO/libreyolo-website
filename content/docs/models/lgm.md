---
title: LGM
families: [lgm]
seo_title: "LGM in Python: one image to 3D Gaussians via four generated views"
description: "Run LGM in LibreYOLO: ImageDream samples four views of an object from one picture, then LGM turns them into 3D Gaussians. MIT model; ImageDream under OpenRAIL use restrictions."
lead: "LGM (Large Multi-View Gaussian Model) turns one picture of an object into 3D Gaussians in two stages: ImageDream samples four views around the object, then LGM reads them and returns the Gaussians. LibreYOLO runs both stages in plain PyTorch through the LibreGen factory."
keywords: [LGM, Large Multi-View Gaussian Model, ImageDream, image to 3D, 3D Gaussian splatting, Gaussian splats, multi-view diffusion, PLY, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("lgm")             # weights download on first use
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # standard 3DGS layout
        print(len(result.gaussians))        # those above opacity 0.005
    - label: CLI
      language: bash
      code: |
        # Saves the object as .ply under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=lgm seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("lgm")
        result = model(SAMPLE_IMAGE, elevation=-10)     # picture taken from above
        result = model(SAMPLE_IMAGE, steps=20, guidance=1.0)   # no guidance
---

## Install

```bash
pip install "libreyolo[gen]"
```

The extra brings `diffusers` and `transformers`, which load ImageDream's VAE,
scheduler and CLIP encoders.

## Predict

Nothing is gated and no login is needed. Weights download on first use at
pinned revisions.

<code-tabs name="predict" />

ImageDream, a multi-view diffusion model fine-tuned from Stable Diffusion
2.1, samples four views around the object from the 256 px input. LGM, an
asymmetric UNet with cross-view attention, reads the four views with their
camera rays and returns one Gaussian per output pixel of each view: 4 x 128 x
128 = 65,536 Gaussians, of which those above opacity 0.005 are kept, as
upstream saves them.

Upstream needs xformers on CUDA and renders with a non-commercial rasterizer.
LibreYOLO runs attention through PyTorch's SDPA and does not render, so the
model runs on NVIDIA GPUs, Apple silicon and CPU, in fp16 on GPUs and fp32 on
CPU. On CUDA a seed reproduces upstream's object; other devices draw other
noise for the same seed.

The output is `result.gaussians`, the same payload as
[Speridlabs Idol](/docs/models/speridlabs-idol), saved as `.ply` or
`.splat`.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `steps` | `30` | ImageDream's DDIM sampling steps. |
| `guidance` | `5.0` | Classifier-free guidance; 1 turns it off. |
| `elevation` | `0` | Camera height of the picture in degrees; negative is above the object. |

## Variants

One size, `0.4b`: the 0.42B LGM network (about 0.8 GB) plus the 1.3B
ImageDream multi-view stage and its two CLIP encoders (about 4.0 GB).

## Export

<export-matrix />

Export is not implemented for LibreGen 3D families. `export()`, `train()` and
`val()` raise `NotImplementedError` (ADR 0031 in the library).

## Licensing

<provenance-box>

LGM is by Jiaxiang Tang et al. at 3D Topia
([3DTopia/LGM](https://github.com/3DTopia/LGM)). The two stages carry
different licences:

- The LGM Gaussian model and its code are MIT.
- **ImageDream carries use restrictions.** It is declared OpenRAIL upstream,
  and LibreYOLO applies the CreativeML OpenRAIL++-M License of its Stable
  Diffusion 2.1 lineage: commercial use is allowed, but its use-based
  restrictions (no illegal, harmful, discriminatory or deceptive uses, among
  others) bind you and must be passed on with the model or derivatives.

Weights are redistributed unchanged at pinned revisions: `ashawkey/LGM` with
BiRefNet (MIT) as LibreYOLO/LibreLGM0.4b, and
`ashawkey/imagedream-ipmv-diffusers` as
[LibreYOLO/LibreImageDream](https://huggingface.co/LibreYOLO/LibreImageDream).
Neither is gated. Upstream's renderer, which needs Inria's non-commercial
rasterizer, is left out.

</provenance-box>

## Citation

<citation-block />
