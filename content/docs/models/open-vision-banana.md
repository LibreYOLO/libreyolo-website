---
title: Open Vision Banana
families: [open_vision_banana]
seo_title: "Open Vision Banana in Python: segmentation, depth and normals from a prompt"
description: "Run Open Vision Banana in LibreYOLO: a FLUX.2 [klein] fine-tune that answers perception tasks with an image. Segmentation, depth, normals, editing and text to image. Non-commercial weights."
lead: "Open Vision Banana (OVB), by FunAILab, is FLUX.2 [klein] Base fine-tuned to answer perception tasks with an image. The prompt names the task and the output image encodes the answer: segmentation as flat colours, depth as grey, surface normals as RGB. The same weights keep image editing and text to image. LibreYOLO runs it through the LibreGen factory in 4B and 9B."
keywords: [Open Vision Banana, OVB, FunAILab, FLUX.2 klein, generative perception, semantic segmentation, depth estimation, surface normals, image editing, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Depth
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("open-vision-banana")      # 4b; "open-vision-banana-9b" for 9b
        DEPTH = (
            "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
            "for every input pixel at the same pixel location, with closer points brighter and farther points "
            "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
            "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
            "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
            "legends, borders, or objects; return only the registered depth map."
        )
        result = model(DEPTH, source=SAMPLE_IMAGE, seed=0)
        result.generated.save("depth.png")
        rgb = result.generated.array                # (H, W, 3) uint8
    - label: Semantic segmentation
      language: python
      code: |
        import json
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("open-vision-banana")
        legend = {"building": [255, 8, 232], "sky": [2, 204, 254], "water": [8, 4, 249]}
        prompt = (
            "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
            f"using this color mapping: {json.dumps(legend, separators=(', ', ': '))} "
            "Color each pixel of a listed class with its color and leave everything else black. "
            "Output only the segmentation visualization image."
        )
        model(prompt, source=SAMPLE_IMAGE, seed=0).generated.save("segmentation.png")
    - label: Text to image
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("open-vision-banana")
        model("an empty tram at a cobblestone stop on a rainy night", imgsz=1024, seed=0).generated.save("tram.png")
    - label: CLI
      language: bash
      code: |
        # The CLI covers text to image only.
        libreyolo generate model=open-vision-banana prompt="an empty tram at a cobblestone stop on a rainy night"
---

## Install

```bash
pip install "libreyolo[gen]"
```

## Predict

The mirrors download once on first use. LibreYOLO loads OVB with diffusers'
`Flux2KleinPipeline` and adds no network code: what you write is the prompt
and a few lines of NumPy to read the colours back.

<code-tabs name="predict" />

`source` takes a path, URL, `PIL.Image` or BGR NumPy array. The output keeps
the photo's aspect ratio, capped at one megapixel; resize it back to the
photo before reading it (nearest-neighbour for masks, so colours stay exact).
Without `source` it is a normal text-to-image model. Knobs: `steps` (28) and
`guidance` (3.5), the paper's settings. `offload=True` keeps idle parts off
the GPU.

The model was trained on many phrasings of each task. The library's
[Open Vision Banana guide](https://github.com/LibreYOLO/libreyolo/blob/v1.8.0/docs/open_vision_banana.md)
has FunAILab's quickstart prompts and readouts for semantic, referring and
instance segmentation, depth, surface normals and image editing, plus legend
ensembling (several passes with shuffled colours and a per-pixel vote).

What to expect: every answer is a 28-step diffusion run, so it is slow. Use it
when one model for many tasks matters more than speed. Masks have soft,
slightly shifted edges after resizing; depth is relative inverse depth up to
an unknown scale and shift, and normals are smooth but not metric.

## Variants

| Size | Alias | Mirror | Text encoder | GPU memory (library figure) |
| --- | --- | --- | --- | --- |
| `4b` | `open-vision-banana` | `LibreYOLO/LibreOpenVisionBanana4b` | Qwen3 4B | about 16 GB |
| `9b` | `open-vision-banana-9b` | `LibreYOLO/LibreOpenVisionBanana9b` | Qwen3 8B | about 34 GB |

Both mirrors hold the paper's non-zero-shot checkpoints, trained with the
benchmarks' training splits. Another OVB checkpoint saved as a diffusers
pipeline (the zero-shot ones) loads with
`LibreGen("open-vision-banana", checkpoint_dir="path/to/folder")`.

## Export

<export-matrix />

Training, export and `val()` are not supported.

## Licensing

<provenance-box>

Open Vision Banana is by FunAILab
([OpenVisionBanana/OpenVisionBanana](https://github.com/OpenVisionBanana/OpenVisionBanana)).
The weights are under Black Forest Labs' **FLUX Non-Commercial License**: no
commercial or production use, and section 2(e) asks for content filtering or
human review of generated images before they are shared. FunAILab's training
code is Apache-2.0, but LibreYOLO ports none of it.

The LibreYOLO mirrors are complete diffusers pipelines, not gated: FunAILab's
transformer cast from float32 to bfloat16, the precision it runs in, and the
rest of BFL's FLUX.2 [klein] Base pipeline byte-identical. Outputs match
FunAILab's own quickstart bit for bit on the same seed.

</provenance-box>

## Citation

<citation-block />
