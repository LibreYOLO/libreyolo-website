---
title: Vision-language API
seo_title: 'LibreVLM API: aliases, set_classes and chat'
description: >-
  The LibreVLM factory, every model alias, the sticky set_classes vocabulary,
  set_task, the chat escape hatch, and why confidence is a placeholder.
lead: >-
  LibreVLM loads a generative vision-language model and drives it as an object
  detector. The class list is a prompt rather than a fixed head, and the model
  returns the same Results any other family returns.
keywords:
  - LibreVLM
  - vision language model detection
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  Aliases read from libreyolo/models/vlm/__init__.py; repositories, sizes and
  task lists from the family modules under libreyolo/models/vlm/ plus
  libreyolo/models/sensenova/model.py; call rules and raises from
  libreyolo/models/vlm/base.py, all at v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Detect an open vocabulary
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Ask a free-form question
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Install

The tier needs the `vlm` extra.

<code-tabs name="install" />

## The factory

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` is an alias, not a path. `**kwargs` reaches the family constructor,
which takes `device`, `names` (the initial vocabulary, equivalent to calling
`set_classes` after load), `prompt` (override the detection prompt) and
`max_new_tokens`. An unknown alias raises `ValueError` listing every alias.

<code-tabs name="usage" />

## Aliases

| Family | Aliases | Sizes | Weights |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Pinned upstream snapshot |

The default alias is `qwen3-vl-4b`. Sizes for the default alias of each family
are the ones listed first: `qwen3-vl` resolves to `4b`, `lfm2-vl` to `450m`,
`internvl3` to `2b`, `smolvlm2` to `2.2b`, `florence-2` to `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` and `LibreMODUS`
(also spelled `LibreModus`) are exported at package level.

## Tasks

Most families serve `detect` only. Two serve more:

| Family | Supported tasks |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Because the task is prompt-driven rather than baked into a checkpoint, it can
be switched on a loaded model:

```python
model.set_task(task: str) -> LibreVLMModel
```

The task is validated against the family's supported list, is sticky across
later `predict()` and `track()` calls, and the model is returned so calls can
chain.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Sets the open vocabulary. Any words work, because the model is prompted with
them rather than constrained to a fixed head. The list must be non-empty and
its entries must be unique when compared case-insensitively. Passing a bare
string raises `TypeError`, because it would enumerate into one-character
classes. The vocabulary is sticky: set it once after loading and it persists
until set again.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Raw multimodal generation: image and prompt in, decoded text out, verbatim.
This is the escape hatch under the detection convenience, for free-form
questions, counting, or an output format the detection wrapper does not
cover. `max_new_tokens` falls back to the family's `MAX_NEW_TOKENS`, which is
1024 on the base class. Decoding is greedy with a mild repetition penalty.

## Confidence

Generated output has no calibrated per-box confidence. This version assigns a
constant placeholder so `predict`, drawing and `track` behave, which makes
`conf=` filtering and mAP soft rather than meaningful. This is also why
`val()` raises: COCO mAP over placeholder scores would mislead.

## Predict and track

The standard predict surface applies, and `track()` works, so a VLM detector
drops into the same pipeline as any other family. Two class-level policies
differ from a convolutional detector: test-time augmentation is disabled,
because multi-scale augmentation is meaningless for a fixed-resolution
generator, and batched predict is off, because generation is autoregressive
and preprocessing returns a text-and-image encoding rather than a stackable
image tensor.

## Not supported

`train()`, `val()` and `export()` raise `NotImplementedError`. Fine-tune
upstream and load the resulting weights.

## Remote code

Every shipped family loads through a native model class, so LibreYOLO does not
execute third-party repository code by default. A family that genuinely needs
it must opt in explicitly and pin a snapshot revision; LocateAnything is the
one that does, pinned to commit `c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS is an explicit exception to the checkpoint schema: its alias
resolves to a directory of pinned upstream files rather than a LibreYOLO
`.pt`, and LibreYOLO neither adds v1.0 metadata to it nor republishes it.


