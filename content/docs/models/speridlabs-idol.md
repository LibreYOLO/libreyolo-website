---
title: Speridlabs Idol
families: [speridlabs_idol]
seo_title: "Speridlabs Idol in Python: one image to 3D Gaussian splats"
description: "Run Speridlabs Idol in LibreYOLO: one picture of an object in, the whole object out as 3D Gaussian splats in .ply or .splat. Apache-2.0, nothing gated."
lead: "Speridlabs Idol is an image-to-3D model by Speridlabs. It takes one picture of an object and returns the whole object, including the sides the picture never showed, as 3D Gaussian splats. LibreYOLO ports it from the Speridlabs SDK and runs it through the LibreGen factory."
keywords: [Speridlabs Idol, Speridlabs, image to 3D, single image to 3D, 3D Gaussian splatting, Gaussian splats, 3DGS, PLY, splat, LibreGen]
last_verified: "1.7.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        # An example image from the Speridlabs Idol SDK. A local path, a PIL
        # image, a NumPy array or a folder of images works the same way.
        IMAGE = "https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp"

        model = LibreGen("speridlabs-idol")   # weights download on first use
        result = model(IMAGE, seed=0)

        result.save("armchair.ply")           # standard 3DGS layout
        result.save("armchair.splat")         # compact web format
        print(len(result.gaussians))          # 262144 at the defaults
    - label: CLI
      language: bash
      code: |
        # Saves the object as .ply under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen

        IMAGE = "https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp"

        model = LibreGen("speridlabs-idol")

        result = model(IMAGE, mode="real")             # photographs: a looser crop
        result = model(IMAGE, gaussians=1_048_576)     # more detail
        result = model(IMAGE, steps=25)                # faster, a little less detail

        # Two objects from one image, seeds 7 and 8, written to runs/generate/predict.
        results = model(IMAGE, seed=7, n=2, save=True)
    - label: Results
      language: python
      code: |
        from libreyolo import LibreGen

        IMAGE = "https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp"

        model = LibreGen("speridlabs-idol")
        result = model(IMAGE)

        g = result.gaussians
        print(g.xyz.shape, g.scales.shape, g.rotations.shape)   # (N, 3), (N, 3), (N, 4) wxyz
        print(g.opacities.shape, g.colors.shape)                 # (N,), (N, 3) RGB
        print(g.camera)                                          # the inferred input view
        print(g.seed, g.params)                                  # how to make it again

        g.input_image.save("crop.png")         # what the model saw after its crop
        result.plot().save("preview.png")      # point-render preview
        g[g.opacities > 0.1].save("opaque.ply")
---

## Install

```bash
pip install "libreyolo[gen]"
```

The `gen` extra installs `diffusers`, `transformers` and `accelerate`.
Speridlabs Idol reads its DINOv3 encoder through `transformers` and its FLUX.2
VAE through `diffusers`. Importing `LibreGen` does not import either; they load
on first use.

## Predict

<code-tabs name="predict" />

The source is anything predict takes: a path, URL, `PIL.Image`, NumPy array
(BGR), tensor, a folder of images, or a list of them. An image file or
`PIL.Image` with real transparency is used as the object's cutout. Any other
input has its background removed first, by LibreYOLO's BiRefNet network with
the BiRefNet weights from the Idol snapshot. The input resolution is fixed, so
`imgsz` raises, and so does `prompt=`: the model has no text input.

The first call downloads about 5 GB at pinned revisions, with nothing gated
and no Hugging Face login: Idol's own weights (about 3.1 GB) from Speridlabs'
repository [speridlabs/idol](https://huggingface.co/speridlabs/idol) into
`weights/LibreSperidlabsIdol0.6b/`, plus two image encoders into the Hugging
Face cache, the DINOv3 ViT-H+/16 encoder (1.7 GB) and the FLUX.2 VAE (0.2 GB).
`LibreGen(path)` loads a local directory that holds the snapshot files
instead.

One source with `n=1` returns a `Results`; several sources or `n > 1` return
a list, and `stream=True` a generator. `result.gaussians` holds the object in
its own frame: z up, centered in the cube `[-0.5, 0.5]^3`. `save()` picks the
format from the extension. A `.ply` is the standard 3D Gaussian Splatting
layout, turned y-up the way splat viewers and trainers expect, with the seed,
camera and settings stored in a header comment. A `.splat` is the compact web
format, 32 bytes per Gaussian. `result.plot()` returns a point render from the
inferred input view, a quick look rather than a Gaussian rasterization.

A seed, with the same model and settings, gives the same object. Object `i`
of a source uses `seed + i`; without a seed, a fresh one is drawn and recorded
in `result.gaussians.seed`. As in the SDK, the seed drives a random generator
on the model's device, so one seed gives different objects on CUDA, MPS and
CPU. On CUDA a seed gives the object the Speridlabs SDK gives for the same seed
and settings. Rarely, part of the Gaussians land differently from one run to
the next while the object stays the same: one attention step runs on a cuDNN
kernel whose result depends on GPU memory history, in the SDK as in the port.

Objects generated from photos, with their source images, are in the article
[Speridlabs Idol in LibreYOLO: one image to a 3D object](/articles/speridlabs-idol-image-to-3d).

### Settings

Every setting has a default, so all are optional. A name the model does not
have raises and lists the supported ones.

| Setting | Default | Meaning |
| --- | --- | --- |
| `steps` | `50` | Sampling steps. Fewer run faster and lose some detail. |
| `guidance` | `3.0` | Classifier-free guidance; `0` turns it off. |
| `shift` | `3.0` | Flow time shift toward high noise. |
| `gaussians` | `262144` | Gaussians in the output, from 32,768 to 1,048,576, rounded down to a multiple of 32. |
| `mode` | `"synthetic"` | `"synthetic"` for renders and generated images, `"real"` for photographs. Sets how much room the crop leaves around the object: a crop side of 1.2 or 1.7 times the object size. |
| `margin` | `None` | An explicit crop side over object size, at least 1. Overrides `mode`. |

The constructor takes `device` (`"auto"` picks CUDA, then MPS, then CPU) and
`compile`. `compile=True` needs CUDA: it compiles the denoiser blocks, runs the
guided and unguided passes in one batch, and uses cuDNN attention and TF32
matmuls while sampling, the same path as the SDK's optimized sampling. It
reproduces that compiled path; against the eager path it gives the same
object, not the same Gaussians. The model runs in its published mixed
precision (fp16 transformer blocks, bf16 encoders), so `dtype` accepts only
`"auto"`.

### Speed and memory

Time per object at the default settings, one object per call, in the
published mixed precision:

| Device | Runtime | Time per object (s) |
| --- | --- | ---: |
| NVIDIA H100 | CUDA, eager | 9.8 |
| NVIDIA H100 | CUDA, `compile=True`, after the first call | 6.4 |
| Apple M4 Pro | MPS, eager | 464 to 467 |

The M4 Pro figures are two objects on one machine, the first including the
model load from local disk. Speridlabs Idol runs in under 8 GB of GPU memory.
Speridlabs publishes request times for its SDK on ten GPUs, from 3.4 s on a
B300 to 54.6 s on an L4, on the
[Idol research page](https://speridlabs.com/research/idol#gpus).

## Export

<export-matrix />

Export is not implemented: the model is a sampling loop over several
networks, and LibreYOLO has no exportable graph contract for that yet (ADR
0031 in the library). `export()`, `train()` and `val()` raise
`NotImplementedError`. `train(data=...)` is reserved for fine-tuning on image
and 3D object pairs, and `val(data=...)` for rendered-view metrics.

## Licensing

<provenance-box>

Speridlabs Idol is the work of Alejandro Cañada and Chema Garabito at
[Speridlabs](https://speridlabs.com). Its code is the
[Idol SDK](https://github.com/speridlabs/idol-sdk) and its weights are
[speridlabs/idol](https://huggingface.co/speridlabs/idol), both Apache-2.0.
LibreYOLO does not mirror those weights: they download from Speridlabs'
repository, so the downloads count for its authors.

The DINOv3 encoder is the one file LibreYOLO hosts for this model, at
[LibreYOLO/LibreDINOv3vith16plus](https://huggingface.co/LibreYOLO/LibreDINOv3vith16plus).
It is Meta's checkpoint cast to bfloat16, the precision Idol runs it in: all
615 tensors equal the upstream ones loaded in bfloat16, so outputs do not
change. Meta's own copy sits behind a manually reviewed access form; the
mirror ships the DINOv3 License agreement and has no gate. The FLUX.2 VAE
downloads from
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
under Apache-2.0. LibreYOLO logs this license summary once per process, when
it loads the weights.

</provenance-box>

## Citation

<citation-block />
