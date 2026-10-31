---
title: libreyolo generate
seo_title: "libreyolo generate command reference"
description: "Generate images from a text prompt, or 3D objects from an image, on the command line: every argument with its default read from the CLI definition."
lead: "Runs a LibreGen model from the shell. With a prompt it generates images, with a source image it generates 3D objects, and either way it saves the output by default."
keywords: [libreyolo generate cli, text to image command line, image to 3D command line, FLUX.2 klein cli, SANA-Sprint cli, Speridlabs Idol cli, libreyolo json output]
last_verified: "1.7.0"
meta:
  - label: Command
    value: libreyolo generate
    mono: true
  - label: Required
    value: "One of prompt, prompt_file or source"
  - label: Output
    value: "One record per image or object on stdout. Files under runs/generate/predict unless save=false"
snippets:
  examples:
    - label: Text to image
      language: bash
      code: |
        # Needs a CUDA GPU or Apple silicon (MPS). The model defaults to flux2-klein-4b.
        libreyolo generate prompt="a red fox sleeping in fresh snow" seed=0
    - label: Image to 3D
      language: bash
      code: |
        # With source= the model defaults to speridlabs-idol; the object is saved as .ply.
        libreyolo generate seed=0 \
          source=https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp
    - label: Prompt file, JSON on stdout
      language: bash
      code: |
        printf 'a red fox sleeping in fresh snow\na lighthouse on a rocky coast\n' > prompts.txt

        # Two images per prompt at 768x1024 (height x width), one JSON object on stdout.
        libreyolo generate prompt_file=prompts.txt model=sana-sprint-0.6b \
          seed=0 n=2 imgsz=768x1024 json=true quiet=true
---

## Synopsis

```bash
libreyolo generate prompt=<text> [model=<alias|dir>] [key=value ...]
libreyolo generate prompt_file=<file> [model=<alias|dir>] [key=value ...]
libreyolo generate source=<image|dir|url> [model=<alias|dir>] [key=value ...]
```

Pass exactly one of `prompt`, `prompt_file` and `source`. Arguments are
`key=value` pairs, and the POSIX form works too, so `seed=0` and `--seed 0`
are interchangeable and `save=false` becomes `--no-save`. Names with an
underscore accept either spelling: `prompt_file=` and `--prompt-file` reach
the same option.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `prompt` | | Text to render. Always literal text, never a path |
| `prompt_file` | | Text file with one prompt per line; blank lines are skipped |
| `source` | | Image, directory of images or URL to turn into 3D objects. Image-to-3D models only |
| `model` | | LibreGen alias or local model directory. `flux2-klein-4b` with a prompt, `speridlabs-idol` with a source |
| `imgsz` | | Output size: `1024` (square) or `768x1024` (HxW). The model's native size when unset. Text-to-image only |
| `seed` | | Seed. The same seed, model and settings repeat the output. A fresh seed is drawn and recorded when unset |
| `n` | `1` | Images per prompt, or objects per source |
| `steps` | | Sampling steps (family setting) |
| `guidance` | | Prompt guidance (family setting) |
| `shift` | | Flow time shift toward high noise (family setting) |
| `gaussians` | | Gaussians per 3D object (family setting) |
| `mode` | | `synthetic` (renders) or `real` (photos): how much room the crop leaves around the object (family setting) |
| `margin` | | Crop side over object size; overrides `mode` (family setting) |
| `device` | `auto` | `auto`, `cpu`, `mps`, or a CUDA device |
| `dtype` | `auto` | `auto`, `bf16`, `fp16` or `fp32`. Image-to-3D models take `auto` only |
| `offload` | `false` | Keep idle components off the GPU. CUDA and text-to-image models only |
| `compile` | `false` | `torch.compile` the 3D sampler. CUDA and image-to-3D models only |
| `save` | `true` | Save images (`.png`) or 3D objects (`.ply`) under `runs/generate/predict` |
| `output_path` | | Single output file: `.png`, or `.ply` or `.splat` for 3D. One prompt or source with `n=1` |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `help_json` | `false` | Dump command schema as JSON and exit |

A family setting left unset takes the model's own default, listed on its page:
[FLUX.2 klein](/docs/models/flux2-klein),
[SANA-Sprint](/docs/models/sana-sprint) and
[Speridlabs Idol](/docs/models/speridlabs-idol). The command forwards only the
settings you pass, and one the model does not have is an error rather than
ignored: `shift=` on a text-to-image model exits with code `2`.

## Examples

<code-tabs name="examples" />

## Notes

Each call saves into a new numbered folder, `runs/generate/predict`, then
`predict2` and so on. An image is named after its index, prompt and seed, and
a 3D object after its index, source image and seed. A PNG also stores prompt,
seed and settings as JSON in a `libreyolo` text chunk, and a `.ply` in a header
comment. `save=false` writes nothing and only prints the records.

The input kind decides the model family. A prompt sent to an image-to-3D
model, a `source` sent to a text-to-image model, and `imgsz` together with
`source` all exit with code `2`. With `prompt_file`, a file holding one prompt
behaves like `prompt=`.

stdout carries results and nothing else; model loading, progress, warnings
and errors go to stderr. `json=true` prints one JSON object per invocation,
with `task`, `model`, `schema_version` and a `results` list holding one record
per image or object and the `path` it was saved to. `quiet=true` silences
stderr, and errors still arrive as JSON on stdout when `json=true` is set.

The exit code is `0` on success, `2` for a usage or configuration error
(including an unknown model alias or a device that is not available), `4`
when the model cannot be loaded (including a missing `libreyolo[gen]` extra)
or its runtime fails, and `1` for an I/O error such as a missing prompt file,
and for running out of CUDA memory.

`help_json=true` prints the command's parameters, types, defaults and flags as
JSON without running anything.

Related: [Image generation](/docs/tasks/image-generation) and
[3D generation](/docs/tasks/3d-generation) for the same call in Python.
