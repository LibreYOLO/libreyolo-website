---
title: Speridlabs Iris-Z
families: [speridlabs_irisz]
seo_title: "Speridlabs Iris-Z in Python: depth, text to image, and depth to image"
description: "Run Speridlabs Iris-Z in LibreYOLO: relative depth from a photo, images from a prompt, an image together with its depth, and a new image painted on a given depth, from one model."
lead: "Speridlabs Iris-Z is a FLUX.2 fine-tune by Speridlabs that denoises an image and its depth in one sequence. The same weights estimate relative depth from a photo, generate an image from a prompt, generate an image together with its depth, and paint a prompt onto a given depth. LibreYOLO ports it from the Speridlabs SDK and runs it through the LibreGen factory."
keywords: [Speridlabs Iris-Z, Iris-Z, Speridlabs, depth estimation, monocular depth, text to image, depth to image, FLUX.2, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Depth
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE, LibreGen

        model = LibreGen("speridlabs-iris-z", task="depth")   # 9b; or speridlabs-iris-z-32b
        result = model(SAMPLE_IMAGE, imgsz=512)

        result.depth_map.data          # (H, W) relative log depth, larger is farther
        result.depth_map.valid_mask    # False on sky
        result.save("depth.png")       # coloured depth
    - label: Text to image
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        result = model("a lighthouse on a cliff in a storm, waves crashing", imgsz=(720, 1280), seed=1)
        result.save("lighthouse.png")
    - label: Image and depth
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        road = model(
            "a photo of a winding mountain road through a pine forest at golden hour, a red car",
            imgsz=(720, 1280),
            seed=1,
            depth=True,
        )
        road.generated.save("road.png")   # the image
        road.depth_map.data               # its relative log depth
    - label: Depth to image
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        road = model("a photo of a winding mountain road through a pine forest, a red car", seed=1, depth=True)

        # A new scene on the same geometry: source= is the depth to paint on.
        snow = model("the same road in deep snow on a bright winter morning", source=road, seed=0)
        snow.save("snow.png")

        # A photo works too: its depth is estimated first.
        from libreyolo import SAMPLE_IMAGE
        model("the same scene at night, city lights", source=SAMPLE_IMAGE).save("night.png")
    - label: CLI
      language: bash
      code: |
        libreyolo generate prompt="a lighthouse at dusk" model=speridlabs-iris-z
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg task=depth imgsz=512
---

## Install

```bash
pip install "libreyolo[gen]"
```

The `gen` extra installs `diffusers`, `transformers` and `accelerate`.
Iris-Z reads the FLUX.2 VAE through `diffusers` and its prompt encoders
through `transformers`. Importing `LibreGen` does not import either; they load
on first use.

## Predict

<code-tabs name="predict" />

The task is fixed when the model is built. `task="depth"` estimates depth from
images and loads no text encoder; the default task, `generate`, takes prompts.

**Depth.** The source is anything predict takes: a path, URL, `PIL.Image`,
NumPy array (BGR), tensor, a folder of images, or a list of them.
`result.depth_map` is relative log depth at the photo's size
(`encoding="log_depth"`, larger is farther), with sky marked invalid in
`valid_mask`. `imgsz` is the working resolution, 512 or 1024; by default 512
for images under half a megapixel and 1024 above. The default seed is 0, so
depth repeats. `ensemble=k` takes the per-pixel median over `k` seeds, each
affinely aligned to the first.

**Text to image.** A prompt, or a list of prompts, returns `Results` with the
image in `result.generated`. `imgsz` is an int for a square image or
`(height, width)`, rounded up to a multiple of 16. Image `i` of a prompt uses
`seed + i`; without a seed, a fresh one is drawn and recorded.

**Image and depth.** `depth=True` samples the image and its depth together,
then estimates the depth again from the finished image, as the Speridlabs SDK
does by default. The result carries both `result.generated` and
`result.depth_map`.

**Depth to image.** `source=` is the depth the image is painted on: a
`Results` or `DepthMap` from any depth model, a photo (its depth is estimated
first, at `imgsz` 512 or 1024, default 1024), or a grayscale depth map read
with `depth_format="depth"` (bright is far) or `"disparity"` (bright is near).
An image is read as a depth map only when `depth_format` says so. With
`source=`, `imgsz` is one side and the output keeps the depth's aspect ratio;
`result.depth_map` is the depth the image was painted on.

### Settings

| Setting | Default | Task | Meaning |
| --- | --- | --- | --- |
| `steps` | `50` | generate | Sampling steps. |
| `guidance` | `4.0` (`9b`), `3.5` (`32b`) | generate | Classifier-free guidance. |
| `depth` | `False` | generate | Also return the image's depth. |
| `steps` | `4` | depth | Depth sampling steps; more than 2. |
| `ensemble` | `1` | depth | Per-pixel median over this many seeds. |

The constructor takes `size` (`"9b"`, Iris-Z-small, the default, or `"32b"`,
Iris-Z-big), `device`, and `offload`. `offload=True` keeps the text encoder on
the CPU between prompts. The model runs in its published bfloat16 precision,
so `dtype` accepts `"auto"` or `"bf16"`.

### Speed and memory

Measured peak GPU memory on an NVIDIA H200: depth 21 GB (`9b`) and 64 GB
(`32b`) at `imgsz=1024`; generation 34 GB and 107 GB with the text encoder.
On the same GPU a depth map takes 0.47 s (`9b`, `imgsz=512`) to 4.8 s
(`32b`, `imgsz=1024`), and a 512x512 image 9.4 s (`9b`) or 30 s (`32b`) at
50 steps. A 1280x720 image from `32b` at 50 steps took 106 to 112 s per image
on an H200, with or without `depth=True` or `source=`.

The first call downloads the size's folder of
[speridlabs/iris-z](https://huggingface.co/speridlabs/iris-z), about 18 GB
(`9b`) or 64 GB (`32b`), plus the FLUX.2 VAE (0.2 GB) and, for `generate`,
the text encoder (9 GB for `9b`, 48 GB for `32b`). Nothing is gated.

## Export

<export-matrix />

Export is not implemented: the model is a sampling loop over several
networks. `export()`, `train()` and `val()` raise `NotImplementedError`.

## Licensing

<provenance-box>

Speridlabs Iris-Z is the work of Alejandro Cañada and Chema Garabito at
[Speridlabs](https://speridlabs.com). Its code is the
[Iris-Z SDK](https://github.com/speridlabs/iris-z-sdk) (Apache-2.0) and its
weights are [speridlabs/iris-z](https://huggingface.co/speridlabs/iris-z),
under the FLUX Non-Commercial License. LibreYOLO does not mirror the weights:
they download from Speridlabs' repository at a pinned revision, so the
downloads count for its authors.

</provenance-box>

## Citation

<citation-block />
