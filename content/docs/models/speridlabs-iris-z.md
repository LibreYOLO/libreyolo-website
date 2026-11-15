---
title: Speridlabs Iris-Z
families: [speridlabs_irisz]
seo_title: "Speridlabs Iris-Z in Python: one model for images and depth"
description: "Run Speridlabs Iris-Z in LibreYOLO: text to image, image to depth, depth to image and image plus depth from one set of FLUX.2-based weights. Non-commercial weights."
lead: "Speridlabs Iris-Z is FLUX.2 fine-tuned to denoise an image and its depth in one sequence. The same weights generate images from a prompt, estimate depth from a photo in 4 steps, paint a prompt onto a given depth, and return an image together with its depth. LibreYOLO runs it through the LibreGen factory in two sizes."
keywords: [Speridlabs Iris-Z, Iris-Z, Speridlabs, FLUX.2, depth estimation, text to image, depth to image, RGB-D generation, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Image to depth
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("speridlabs-iris-z", task="depth")   # 9b; loads no text encoder
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        print(depth.encoding)          # "log_depth": larger is farther
        print(depth.data.shape)        # (H, W) at the photo's size
        print(depth.valid_mask.shape)  # False on sky
        result.save("depth.png")       # coloured depth
    - label: Text to image
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        result = model("a lighthouse at dusk", seed=0)
        result.generated.save("lighthouse.png")
    - label: Depth to image
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("speridlabs-iris-z")
        # The photo's depth is estimated first, then a new scene is painted on it.
        result = model("a sunken ruin under the sea", source=SAMPLE_IMAGE, seed=0)
        result.generated.save("ruin.png")
    - label: Image plus depth
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        result = model("a grand old library, a spiral staircase", depth=True, seed=0)
        result.generated.save("library.png")
        print(result.depth_map.data.shape)
    - label: CLI
      language: bash
      code: |
        libreyolo generate prompt="a lighthouse at dusk" model=speridlabs-iris-z
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg task=depth
        libreyolo generate source=photos/ task=depth ensemble=10 imgsz=1024
---

## Install

```bash
pip install "libreyolo[gen]"
```

## Predict

The task is fixed when the model loads: `task="depth"` for image to depth,
the default `generate` for the other three modes. A depth model never loads a
text encoder.

<code-tabs name="predict" />

**Image to depth.** Sources are what predict takes. `result.depth_map` is a
`DepthMap` with `encoding="log_depth"` (relative log depth, larger is
farther) at the source's size, and `valid_mask` is `False` on sky. `imgsz` is
the working resolution, 512 or 1024 (default 512 under half a megapixel, else
1024). Knobs: `steps` (4) and `ensemble` (1), the per-pixel median over that
many seeds. The default seed is 0, so depth repeats. `save=True` writes the
coloured map under `runs/depth/predict`.

**Text to image.** Prompts, `imgsz`, `seed` and `n` work as for every
[image generation](/docs/tasks/image-generation) model. Knobs: `steps` (50)
and `guidance` (4.0 for `9b`, 3.5 for `32b`).

**Depth to image.** `source=` on the generate task is the depth the image is
painted on: a photo (its depth is estimated first), a `Results` or `DepthMap`
from any depth model, or a grayscale depth image with `depth_format="depth"`
(bright is far) or `"disparity"` (bright is near). An image is only read as a
depth map when `depth_format` says so. The output keeps the depth's aspect
ratio, and `Results.depth_map` is the depth it was painted on.

**Image plus depth.** `depth=True` samples image and depth jointly, then
re-estimates the depth from the finished image, as upstream does by default.
The result carries both `result.generated` and `result.depth_map`.

The CLI covers text to image and image to depth; `prompt=` and `source=` stay
mutually exclusive there.

## Variants

| Size | Alias | Parameters | Text encoder | GPU memory (library figure) |
| --- | --- | --- | --- | --- |
| `9b` | `speridlabs-iris-z`, `-9b`, `-small` | 9.1B, from FLUX.2 [klein] Base 9B | Qwen3 8B FP8 | about 18 GB, plus 9 GB text encoder and 0.2 GB VAE |
| `32b` | `speridlabs-iris-z-32b`, `-big` | 32.2B, from FLUX.2 [dev] | Mistral Small 3.2 24B | about 64 GB, plus 48 GB text encoder and 0.2 GB VAE |

`offload=True` keeps the text encoder on the CPU between prompts. On CUDA
without the FP8 kernel that transformers asks for (the `kernels` package),
and on MPS or CPU, the `9b` text encoder runs dequantized in bfloat16, close
to but not identical with upstream's embeddings. A local snapshot or variant
folder loads with `LibreGen(path)`.

## Export

<export-matrix />

`export()`, `train()` and `val()` raise, as for every LibreGen family.

## Licensing

<provenance-box>

Speridlabs Iris-Z is by [Speridlabs](https://speridlabs.com/research/iris-z).
The weights are under Black Forest Labs' **FLUX Non-Commercial License**: no
commercial or production use, and section 2(e) asks for content filtering or
human review of generated images before they are shared. LibreYOLO's adapter
is MIT; the transformer, samplers and prompt embedders ported from the
[Iris-Z SDK](https://github.com/speridlabs/iris-z-sdk) keep Apache-2.0.

LibreYOLO does not redistribute any weights. They download at runtime,
unmodified, from `speridlabs/iris-z` at a pinned revision, not gated. The
FLUX.2 VAE comes from the Apache-2.0 `black-forest-labs/FLUX.2-klein-4B`
repository, and the text encoders (`Qwen/Qwen3-8B-FP8` for `9b`,
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` for `32b`) are Apache-2.0.

</provenance-box>

## Citation

<citation-block />
