---
title: SANA-Sprint
families: [sana_sprint]
seo_title: "SANA-Sprint in Python: text to image in two steps"
description: "Run SANA-Sprint 0.6B by NVIDIA in LibreYOLO: a text prompt in, a 1024 px image out in two sampling steps, with the seed that makes it again. Apache-2.0."
lead: "SANA-Sprint 0.6B is a text-to-image model by NVIDIA: a 0.6-billion-parameter linear diffusion transformer distilled to sample in one to four steps, with a Gemma 2 2B text encoder. LibreYOLO runs it through its diffusers pipeline behind the LibreGen factory."
keywords: [SANA-Sprint, Sana Sprint 0.6B, NVIDIA, text to image python, image generation, diffusers, SanaSprintPipeline, LibreGen]
last_verified: "1.7.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        # Needs a CUDA GPU or Apple silicon (MPS); generation is not practical on CPU.
        model = LibreGen("sana-sprint-0.6b")   # weights download on first use
        result = model("a red fox sleeping in fresh snow", seed=0)

        result.save("fox.png")                 # the PNG also records prompt, seed and settings
        print(result.generated.array.shape)    # (1024, 1024, 3)
    - label: CLI
      language: bash
      code: |
        # Saves four images as .png under runs/generate/predict.
        libreyolo generate prompt="a red fox sleeping in fresh snow" model=sana-sprint-0.6b seed=0 n=4
    - label: Settings
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("sana-sprint-0.6b")

        result = model("a red fox sleeping in fresh snow", imgsz=(768, 1024))  # height, width
        result = model("a red fox sleeping in fresh snow", steps=4)            # more sampling steps
        result = model("a red fox sleeping in fresh snow", guidance=3.0)

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

        model = LibreGen("sana-sprint-0.6b")
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
SANA-Sprint loads as the diffusers `SanaSprintPipeline`, which needs
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

The first call downloads about 7.7 GB at a pinned revision, with nothing gated
and no Hugging Face login, from
[Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers)
into `weights/LibreSanaSprint0.6b/`. Only the pipeline's component folders
are fetched. `LibreGen(path)` loads a local directory that holds a
`SanaSprintPipeline` in the diffusers layout instead, such as an offline copy
or a fine-tune made with upstream tooling. `sana-sprint` is an alias of
`sana-sprint-0.6b`.

A seed, with the same model and settings, gives the same image. Image `i` of a
prompt uses `seed + i`, and every prompt in a list starts from the same seed.
Without a seed, a fresh one is drawn for each prompt and recorded in
`result.generated.seed`. The seed drives a random generator on the CPU, so the
starting noise is the same on every device; the image can still differ across
devices and precisions by numerical error.

Image-conditioned generation is not available yet: `source=` raises
`NotImplementedError`.

### Settings

Every setting has a default, so all are optional. A name the model does not
have raises `TypeError` and lists the supported ones.

| Setting | Default | Meaning |
| --- | --- | --- |
| `imgsz` | `1024` | Output size: an int for a square image, or `(height, width)`. Each side is rounded up to a multiple of 32, with a warning. |
| `steps` | `2` | Sampling steps. The model is distilled for 1 to 4. At `2` the pipeline uses its own intermediate timestep; any other count runs an evenly spaced schedule. |
| `guidance` | `4.5` | Prompt guidance. |

The constructor takes `device`, `dtype` and `offload`. `dtype="auto"` is bf16
on CUDA (fp16 on a GPU without bf16), bf16 on MPS from macOS 14, and fp32 on
the CPU and on older macOS; `"bf16"`, `"fp16"` and `"fp32"` force one.
`offload=True` keeps idle pipeline components off the GPU; it needs CUDA and
raises on any other device. `model.pipeline` is the loaded diffusers
`SanaSprintPipeline`, for anything the LibreGen call does not cover.

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

SANA-Sprint is the work of NVIDIA. Its code is
[NVlabs/Sana](https://github.com/NVlabs/Sana) and its weights are
[Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers),
both Apache-2.0. The pipeline bundles Google's Gemma 2 2B as its text encoder,
which stays under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
LibreYOLO adds no content filter; the upstream usage terms apply.

</provenance-box>

## Citation

<citation-block />
