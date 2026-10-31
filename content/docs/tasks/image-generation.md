---
title: Image generation
seo_title: "Text-to-image generation in Python with LibreYOLO"
description: "Generate images from text in LibreYOLO with FLUX.2 klein or SANA-Sprint: the LibreGen call, seeds that reproduce an image, settings, saving and the CLI."
lead: "Image generation turns a text prompt into a synthesized image. LibreYOLO exposes it as the generate task, served by the LibreGen factory: one prompt in, one RGB image out, together with the seed and settings that produce it again."
keywords: [text to image python, image generation python, generate images from text, FLUX.2 klein, SANA-Sprint, diffusers, LibreGen, libreyolo generate task]
last_verified: "1.7.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        # Needs a CUDA GPU or Apple silicon (MPS); generation is not practical on CPU.
        model = LibreGen()   # FLUX.2 klein 4B, the default; weights download on first use
        result = model("a red fox sleeping in fresh snow", seed=0)

        result.save("fox.png")              # the PNG also records prompt, seed and settings
        print(result.generated.seed, result.generated.params)
    - label: Several images
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("sana-sprint-0.6b")

        # Two prompts, two images each (seeds 0 and 1), written to runs/generate/predict.
        results = model(
            ["a red fox sleeping in fresh snow", "a lighthouse on a rocky coast"],
            seed=0,
            n=2,
            imgsz=(768, 1024),   # height, width
            save=True,
        )
        for result in results:
            print(result.generated.prompt, result.generated.seed, result.saved_path)
    - label: CLI
      language: bash
      code: |
        # Saves the image as .png under runs/generate/predict.
        libreyolo generate prompt="a red fox sleeping in fresh snow" seed=0
---

## Definition

The `generate` task returns one synthesized image per result, in
`result.generated` as a `GeneratedImage`. `array` is the `(H, W, 3)` uint8 RGB
image and `image` the same image as a `PIL.Image`. `prompt` is the text that
produced it, `seed` the integer that reproduces it with the same model and
settings, and `params` the rest of the call: the model, `imgsz` and every
family setting in effect, defaults included.

A generated image has no source canvas: `result.boxes` and `result.path` are
`None`, `result.orig_shape` is the image's own `(H, W)`, and `len(result)` is
1. `result.save(path)` writes the image, and a PNG also stores prompt, seed and
params as JSON in a `libreyolo` text chunk. `result.plot()` returns the image,
and `result.summary()` one row with the payload's record. The full payload is
described in [Results types](/docs/reference/results-types#generatedimage).

The canonical task key is `generate`. `generation`, `image-generation`,
`text-to-image`, `text2image`, `txt2img` and `t2i` normalize to it.

## Models

Two families serve this task, both through `LibreGen` and both loaded as
`diffusers` pipelines at pinned revisions.
[FLUX.2 klein](/docs/models/flux2-klein) 4B, by Black Forest Labs, is the
default: `LibreGen()` with no argument loads it.
[SANA-Sprint](/docs/models/sana-sprint) 0.6B, by NVIDIA, is the smaller
model. Both are Apache-2.0; SANA-Sprint's bundled Gemma 2 text encoder is
under the Gemma Terms of Use.

`LibreGen` also loads a local directory that holds either family's pipeline in
the diffusers layout, matched by the pipeline class recorded in its
`model_index.json`: an offline copy, or a fine-tune made with upstream
tooling. `LibreYOLO()` does not load these models.

## Predict

<code-tabs name="predict" />

`model(...)`, `model.predict(...)` and `model.generate(...)` are the same
call. The prompt is a string or a list of strings, and a string is always a
prompt, never a path. One prompt with `n=1` returns a `Results`; a list of
prompts or `n > 1` returns a flat list in prompt order, and `stream=True` a
generator.

Every model takes the same arguments. `imgsz` is an int for a square image or
`(height, width)`, defaulting to the model's native size, and each side is
rounded up to the model's size multiple with a warning. `n` is the number of
images per prompt. `save=True` writes each image as a PNG to a new
`runs/generate/predict` folder per call; `output_path` names the file instead,
for one prompt with `n=1`.

A seed, with the same model and settings, gives the same image. Image `i` of a
prompt uses `seed + i`, and every prompt in a list starts from the same seed,
so a prompt and a seed name one image wherever the prompt sits in the list.
Without a seed, a fresh one is drawn for each prompt and recorded in
`result.generated.seed`. Seeds drive a random generator on the CPU, so the
starting noise does not depend on the device; the image can still differ
across devices and precisions by numerical error.

Settings that only some models have, such as `steps` and `guidance`, are
family settings. Each family has defaults, so a bare `model(prompt)` always
works, and a name the family does not have raises `TypeError` listing the
supported ones. Each model page lists its settings and defaults.

Generation runs on a CUDA GPU or on Apple silicon through MPS. `device="auto"`
picks CUDA, then MPS, then the CPU, where the pipeline loads in fp32 and image
generation is not practical. On CUDA, `LibreGen(name, offload=True)` keeps
idle pipeline components off the GPU.

Image-conditioned generation (editing, image to image) is reserved:
`source=` raises `NotImplementedError` on both families. The
[`libreyolo generate`](/docs/cli/generate) command runs the same call from the
shell and saves by default.

## Train

No family in this task trains inside LibreYOLO: `train()` raises
`NotImplementedError`, and `train(data=...)` is reserved for fine-tuning on
captioned images.

## Validate

There is no generation metric yet: `val()` raises `NotImplementedError`, and
`val(data=...)` is reserved for metrics on a captioned image set.

## Export

Export is not implemented. A sampling loop over several networks has no
exportable graph contract yet, so `export()` raises `NotImplementedError`.
