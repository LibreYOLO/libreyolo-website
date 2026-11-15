---
title: PRX
families: [prx]
seo_title: "PRX and PRX Pixel in Python: Photoroom's open text-to-image models"
description: "Run Photoroom's PRX and PRX Pixel text-to-image models in LibreYOLO through LibreGen. Nine PRX checkpoints at 256, 512 and 1024 plus the 7B PRX Pixel. Apache-2.0."
lead: "PRX is Photoroom's open-source text-to-image model, a 1.3B transformer released as nine checkpoints at 256, 512 and 1024 px. PRX Pixel is a 7B sibling that denoises pixels directly, with no autoencoder. LibreYOLO runs all ten through the LibreGen factory on the diffusers pipelines."
keywords: [PRX, PRX Pixel, PRXPixel, Photoroom, text to image, image generation, diffusion, DC-AE, T5Gemma, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("prx-512-sft")
        result = model("a red fox sleeping in fresh snow", seed=0)
        result.save("fox.png")             # the PNG also stores prompt, seed and params
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("prx-512-sft-distilled")          # 8-step distillation
        result = model("a red fox", steps=8, guidance=1.0)

        model = LibreGen("prx")                            # prx-1024-beta
        result = model("a red fox", negative="blurry", imgsz=1024, seed=0)

        model = LibreGen("prxpixel", offload=True)         # prxpixel-7b, needs diffusers>=0.39
        results = model(["a red fox", "a blue jay"], n=2, save=True)
    - label: CLI
      language: bash
      code: |
        libreyolo generate prompt="a red fox" model=prx-512-sft negative="blurry"
        libreyolo generate prompt="a red fox" model=prxpixel-7b seed=0 n=4
---

## Install

```bash
pip install "libreyolo[gen]"
```

PRX Pixel needs `diffusers>=0.39` and says so when it loads; the extra's
floor is 0.38 so it can sit next to the Marigold extra.

## Generate

Weights download from LibreYOLO's byte-identical mirrors of Photoroom's
repositories on first use.

<code-tabs name="predict" />

A string is always a prompt. One prompt with `n=1` returns a `Results`; a
list of prompts or `n > 1` returns a flat list, and `stream=True` a
generator. `result.generated` holds the image (`array`, `image`), the
`prompt`, the `seed` that reproduces it and the `params` in effect. Every
family knob is optional; one the family does not have raises and lists the
supported ones. See the [image generation](/docs/tasks/image-generation) task
page for prompts, seeds, devices and offload.

## Variants

| Model | Parameters | Knobs (defaults) | GPU memory (library figure) |
| --- | --- | --- | --- |
| `prx-1024-beta` (alias `prx`) | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 21 GB |
| `prx-512-sft-distilled` | 1.3B | `steps` (8), `guidance` (1.0), `negative` ("") | about 16 GB |
| `prx-512-sft`, `prx-512` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 16 GB |
| `prx-512-dc-ae-sft-distilled` | 1.3B | `steps` (8), `guidance` (1.0), `negative` ("") | about 16 GB |
| `prx-512-dc-ae-sft`, `prx-512-dc-ae` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 16 GB |
| `prx-256-sft`, `prx-256` | 1.3B | `steps` (28), `guidance` (5.0), `negative` ("") | about 16 GB |
| `prxpixel-7b` (alias `prxpixel`) | 7B | `steps` (28), `guidance` (5.0), `negative` ("") | about 18 GB |

The number is the native resolution. `-dc-ae` uses the 32x DC-AE
autoencoder instead of the FLUX.1 VAE, so its sides round up to a multiple of
32 instead of 16. `-sft` is the supervised fine-tune, ready to use, and
`-sft-distilled` the 8-step distillation of it. Checkpoints without `-sft` are
base models meant for fine-tuning. `prx-1024-beta` is Photoroom's 1024
preview. PRX Pixel bundles a Qwen3-VL text encoder and has no autoencoder.

## Export

<export-matrix />

`export()`, `train()` and `val()` raise, as for every LibreGen family.

## Licensing

<provenance-box>

PRX and PRX Pixel are by [Photoroom](https://github.com/Photoroom/PRX). The
weights are Apache-2.0, so commercial use is fine. The exception is PRX's
bundled T5Gemma 2B text encoder, which stays under Google's **Gemma Terms of
Use** and Prohibited Use Policy: commercial use is allowed, but the
prohibited uses pass on to anyone you share it with, and PRX's model cards
forbid them for the whole model. The autoencoder is the FLUX.1 [schnell] VAE
(Apache-2.0) or DC-AE (MIT); PRX Pixel's Qwen3-VL-Embedding-2B text encoder
is Apache-2.0.

LibreYOLO's adapter is its own MIT code on the Apache-2.0 diffusers
`PRXPipeline` and `PRXPixelPipeline`. Weights download from
`LibreYOLO/LibrePRX<size>` and `LibreYOLO/LibrePRXPixel7b`, byte-identical
mirrors pinned to a commit, not gated.

</provenance-box>

## Citation

<citation-block />
