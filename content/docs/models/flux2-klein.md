---
title: FLUX.2 klein
families: [flux2_klein]
seo_title: "FLUX.2 klein in Python: text to image with LibreYOLO"
description: "Run FLUX.2 klein 4B by Black Forest Labs in LibreYOLO: a text prompt in, a 1024 px image out, with the seed that makes it again. Apache-2.0, nothing gated."
lead: "FLUX.2 klein 4B is a text-to-image model by Black Forest Labs: a 4-billion-parameter rectified-flow transformer, step-distilled to sample in four steps. LibreYOLO runs it through its diffusers pipeline behind the LibreGen factory, where it is the default model."
keywords: [FLUX.2 klein, "FLUX.2 [klein] 4B", Black Forest Labs, text to image python, image generation, diffusers, Flux2KleinPipeline, LibreGen]
last_verified: "1.7.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        # Needs a CUDA GPU or Apple silicon (MPS); generation is not practical on CPU.
        model = LibreGen("flux2-klein-4b")   # weights download on first use
        result = model("a red fox sleeping in fresh snow", seed=0)

        result.save("fox.png")               # the PNG also records prompt, seed and settings
        print(result.generated.array.shape)  # (1024, 1024, 3)
    - label: CLI
      language: bash
      code: |
        # Saves the image as .png under runs/generate/predict.
        libreyolo generate prompt="a red fox sleeping in fresh snow" model=flux2-klein-4b seed=0
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("flux2-klein-4b")

        result = model("a red fox sleeping in fresh snow", imgsz=(768, 1024))  # height, width
        result = model("a red fox sleeping in fresh snow", steps=8)            # more sampling steps

        # Two prompts, two images each (seeds 0 and 1), written to runs/generate/predict.
        results = model(
            ["a red fox sleeping in fresh snow", "a lighthouse on a rocky coast"],
            seed=0,
            n=2,
            save=True,
        )
    - label: Results
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("flux2-klein-4b")
        result = model("a red fox sleeping in fresh snow", seed=0)

        g = result.generated
        print(g.array.shape, g.array.dtype)   # (1024, 1024, 3) uint8, RGB
        print(g.prompt, g.seed)               # what to pass to make it again
        print(g.params)                       # model, imgsz, steps, guidance

        g.image.save("fox.jpg")               # a PIL image
        print(result.summary())               # one JSON-ready row
---

## Install

```bash
pip install "libreyolo[gen]"
```

The `gen` extra installs `diffusers`, `transformers` and `accelerate`.
FLUX.2 klein loads as the diffusers `Flux2KleinPipeline`, which needs
`diffusers` 0.38.0 or later. Importing `LibreGen` imports none of them; they
load on first use.

Generation runs on a CUDA GPU or on Apple silicon through MPS.
`device="auto"` picks CUDA, then MPS, then the CPU, where the pipeline loads
in fp32 and image generation is not practical.

## Predict

<code-tabs name="predict" />

One prompt with `n=1` returns a `Results`; a list of prompts or `n > 1`
returns a flat list in prompt order, and `stream=True` a generator. A string
is always a prompt, never a path. `result.generated` is a `GeneratedImage`:
`array` is the `(H, W, 3)` uint8 RGB image and `image` the same image as a
`PIL.Image`, `prompt` the text, `seed` the integer that reproduces it, and
`params` the model, `imgsz` and every setting in effect, defaults included.
`result.save(path)` writes the image, and a PNG also stores prompt, seed and
params as JSON in a `libreyolo` text chunk. `result.plot()` returns the image.
`save=True` writes each image to a new `runs/generate/predict` folder per
call, the file named after its index, prompt and seed; `output_path` names the
file instead, for one prompt with `n=1`.

The first call downloads about 16 GB at a pinned revision, with nothing gated
and no Hugging Face login, from Black Forest Labs' repository
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
into `weights/LibreFlux2Klein4b/`. Only the pipeline's component folders are
fetched, not the repository's single-file checkpoint. `LibreGen(path)` loads a
local directory that holds a `Flux2KleinPipeline` in the diffusers layout
instead, such as an offline copy or a fine-tune made with upstream tooling.
`flux2-klein`, `flux.2-klein` and `flux.2-klein-4b` are aliases of
`flux2-klein-4b`, and `LibreGen()` with no argument loads it.

A seed, with the same model and settings, gives the same image. Image `i` of a
prompt uses `seed + i`, and every prompt in a list starts from the same seed.
Without a seed, a fresh one is drawn for each prompt and recorded in
`result.generated.seed`. The seed drives a random generator on the CPU, so the
starting noise is the same on every device; the image can still differ across
devices and precisions by numerical error.

Image-conditioned generation, which FLUX.2 klein supports upstream, is not
available yet: `source=` raises `NotImplementedError`.

### Settings

Every setting has a default, so all are optional. A name the model does not
have raises `TypeError` and lists the supported ones.

| Setting | Default | Meaning |
| --- | --- | --- |
| `imgsz` | `1024` | Output size: an int for a square image, or `(height, width)`. Each side is rounded up to a multiple of 16, with a warning. |
| `steps` | `4` | Sampling steps. The distilled checkpoint is trained for 4. |
| `guidance` | `1.0` | Fixed. The checkpoint is step-distilled and runs without guidance, so any other value raises `ValueError`. |

The constructor takes `device`, `dtype` and `offload`. `dtype="auto"` is bf16
on CUDA (fp16 on a GPU without bf16), bf16 on MPS from macOS 14, and fp32 on
the CPU and on older macOS; `"bf16"`, `"fp16"` and `"fp32"` force one.
`offload=True` keeps idle pipeline components off the GPU; it needs CUDA and
raises on any other device. `model.pipeline` is the loaded diffusers
`Flux2KleinPipeline`, for anything the LibreGen call does not cover.

## Export

<export-matrix />

Export is not implemented: the model is a sampling loop over several
networks, and LibreYOLO has no exportable graph contract for that yet (ADR
0030 in the library). `export()`, `train()` and `val()` raise
`NotImplementedError`. `train(data=...)` is reserved for fine-tuning on
captioned images, and `val(data=...)` for generation metrics on a captioned
image set.

## Licensing

<provenance-box>

FLUX.2 klein is the work of [Black Forest Labs](https://bfl.ai). Its reference
code is [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)
and its weights are
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
both Apache-2.0. The same repository supplies the FLUX.2 VAE that
[Speridlabs Idol](/docs/models/speridlabs-idol) reads its input through.
LibreYOLO adds no content filter; the model card lists the uses Black Forest
Labs excludes.

</provenance-box>

## Citation

<citation-block />
