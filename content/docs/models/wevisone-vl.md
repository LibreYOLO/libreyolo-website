---
title: WeVisOne-VL
families: [wevisonevl]
seo_title: "WeVisOne-VL in LibreYOLO: detect, segment, matte, OCR"
description: "Use Tencent's WeVisOne-VL in LibreYOLO for open-vocabulary and referring detection, points, masks, alpha mattes and OCR from one Apache-2.0 VLM."
lead: "WeVisOne-VL is a vision-language model from WeChat Vision at Tencent, fine-tuned from Qwen3-VL to answer perception requests as tagged text, with a small pixel decoder that turns its boxes into masks and alpha mattes. LibreYOLO loads it through LibreVLM and serves five tasks from one checkpoint: detect, point, segment, matte and ocr."
keywords: [WeVisOne-VL, Tencent, WeChat Vision, vision-language model, open-vocabulary detection, referring expression, alpha matting, referring segmentation, OCR, LibreVLM]
last_verified: "1.7.0"
snippets:
  predict:
    - label: Detect a vocabulary
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # The vocabulary is sticky: it holds for every later predict() until
        # set again. detect asks for the whole list in one generation.
        model = LibreVLM("wevisone-vl", names=["person", "railing"])
        result = model.predict(SAMPLE_IMAGE)

        for cls, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
            print(result.names[int(cls)], xyxy.tolist())
    - label: Referring detection
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # referring=True sends one generation per phrase. A phrase the model
        # finds nothing for returns no box.
        model = LibreVLM(
            "wevisone-vl",
            names=["the person wearing white shoes", "a red car"],
            referring=True,
        )
        result = model.predict(SAMPLE_IMAGE)

        for cls, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
            print(result.names[int(cls)], xyxy.tolist())
    - label: Points
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # point sends one generation per vocabulary entry.
        model = LibreVLM("wevisone-vl", names=["person"], task="point")
        result = model.predict(SAMPLE_IMAGE)

        print(len(result.points))   # how many were found
        print(result.points.xy)     # (N, 2) in original-image pixels
    - label: Masks and mattes
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # segment and matte read each vocabulary entry as a phrase, so they
        # raise without one.
        model = LibreVLM("wevisone-vl", names=["person"], task="segment")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy)
        print(result.masks.data.shape)   # (N, H, W) binary masks

        # set_task() switches tasks on the loaded weights, without reloading.
        model.set_task("matte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.matte.array.shape)  # (H, W) float32 alpha in [0, 1]
        result.save("people.png")        # transparent PNG cutout
    - label: OCR
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # ocr ignores the vocabulary. SAMPLE_IMAGE keeps this runnable; point
        # it at an image with text of your own.
        model = LibreVLM("wevisone-vl", task="ocr")
        result = model.predict(SAMPLE_IMAGE)

        for text, quad in zip(result.ocr.texts, result.ocr.data):
            print(repr(text), quad.tolist())
    - label: Raw chat
      language: python
      code: |
        from libreyolo import LibreWeVisOneVL, SAMPLE_IMAGE

        model = LibreWeVisOneVL(size="2b")

        # chat() returns the answer verbatim, tags included: boxes and points
        # are bins on a 0-1000 grid over the image.
        text = model.chat(SAMPLE_IMAGE, "point to: person")
        print(text)
---

## Install

WeVisOne-VL needs the `vlm` extra, which pulls in `transformers`. The
backbone loads through the native Qwen3-VL classes in `transformers`, so no
code from the model repository runs.

```bash
pip install "libreyolo[vlm]"
```

## Predict

`LibreWeVisOneVL` is a Python class, not a `.pt` checkpoint: the `LibreYOLO()`
factory and the `libreyolo` CLI do not resolve it. Load it through
`LibreVLM("wevisone-vl")`, which constructs that class, or import
`LibreWeVisOneVL` directly. The first call downloads the weights from
Tencent's own Hugging Face repositories, pinned to one commit per size, into
`weights/LibreWeVisOneVL2b/` (or `4b`) under the current working directory.
The download includes `mask_decoder.pt`, the pixel decoder that the `segment`
and `matte` tasks run, and the pin keeps it on the same snapshot as the
backbone it was trained against.

`device="auto"`, the default, picks CUDA, then Apple MPS, then CPU. On CUDA
the backbone loads in bf16, or fp16 on a GPU without bf16 support; on MPS and
CPU it loads in fp32. The pixel decoder always runs in fp32.

<code-tabs name="predict" />

The five tasks share one loaded checkpoint, and `set_task()` switches between
them without reloading. Each task fills a different `Results` payload and
sends a different number of generations per image:

| Task | Generations per image | Fills |
|---|---|---|
| `detect` | One for the whole vocabulary; with `referring=True`, one per phrase | `result.boxes` |
| `point` | One per vocabulary entry | `result.points` |
| `segment` | One per phrase, then one more backbone pass per phrase for the pixel decoder | `result.boxes` and binary `result.masks` |
| `matte` | As `segment` | `result.matte`, the pixelwise maximum of every instance matte |
| `ocr` | One, with the vocabulary ignored | `result.ocr` |

`names=` at construction calls `set_classes()` for you. Left unset, the
vocabulary is the COCO-80 names, so `point` sends 80 generations per image.
`segment`, `matte` and `referring=True` detection treat each entry as a
phrase, such as `"the dog on the left"`, and raise `ValueError` at predict
time when no vocabulary was set. With `referring=True` each phrase gets the
referring instruction, which the model was tuned to answer with `None` when
nothing matches; a `None` answer returns no box for that phrase.

`segment` thresholds each instance matte at `mask_threshold` (default 0.5) to
get its binary mask. `decoder_batch_size` (default 1) sets how many instances
the pixel decoder refines per call. It bounds memory and does not change the
masks: refinement runs on a 1024x1024 canvas, at about 2 GB of fp32
activations per instance by the library's own figure. `matte` writes a
transparent PNG through `result.save()`, and `result.cutout()` returns the
same RGBA array in memory. For `ocr`, the released instruction asks for every
word in box format, and each box comes back as an axis-aligned quad with its
transcript, in reading order.

Confidence is a constant 1.0 on every box, point and OCR region, because
WeVisOne-VL emits no score: `conf=` only drops rows below that constant, it
does not rank them. On `detect` and `segment`, `iou` drops a later box of the
same class once it overlaps a kept one past the threshold. `chat()` returns
the raw answer with its coordinate tags, which is the route for the upstream
abilities LibreYOLO does not wrap as tasks, such as visual-prompt detection
and OCR grounding. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

Two sizes, fine-tuned from Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct.
`LibreVLM("wevisone-vl")` loads 2B, as do the aliases `wevisone`, `wevisonevl`
and `wevisone-vl-2b`; `LibreVLM("wevisone-vl-4b")` loads 4B, and
`LibreWeVisOneVL(size="4b")` is the class form. Both declare a nominal 1440 px
input, but the Qwen processor resizes each image within a pixel budget set by
`min_pixels` and `max_pixels` (16x32x32 to 2025x32x32 pixels by default), so
that figure is not a fixed operating resolution. Lowering `max_pixels` shrinks
the image the backbone sees, trading accuracy for speed.

Tencent reports per-task results for both sizes in the upstream README, and
LibreYOLO has not reproduced them. LibreYOLO's recorded check ran the 2B
checkpoint through all five tasks on CPU in fp32, and the ported pixel decoder
matches the upstream module's logits to within 1e-4. The 4B decoder weights
load against the 4B configuration, but the 4B backbone has not been run end to
end, and no CUDA run is recorded for either size.

LibreYOLO exposes this family for prediction only. `train()` raises
`NotImplementedError` because the upstream release ships no training recipe
for the backbone or the pixel decoder; Qwen3-VL is the VLM that trains, see
[VLM fine-tuning](/docs/train/vlm-fine-tuning). `val()` raises because a
constant confidence would make COCO mAP misleading, and `export()` raises for
every format.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
