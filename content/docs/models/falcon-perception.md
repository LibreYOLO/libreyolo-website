---
title: Falcon Perception
families: [falconperception]
seo_title: "Falcon Perception in Python: text-query detection and segmentation"
description: "Run TII's Falcon Perception in LibreYOLO: find every instance of a text query and return boxes, or boxes plus masks on 600m. Apache-2.0."
lead: "Falcon Perception is an early-fusion transformer by the Technology Innovation Institute (TII) that reads an image and a natural-language query and returns every matching instance. LibreYOLO vendors its inference network for open-vocabulary detection on both sizes and instance segmentation on 600m."
keywords: [Falcon Perception, TII, open-vocabulary detection, referring expression, text-prompted segmentation, grounding, early fusion, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # Each vocabulary entry is one query; instances get that entry's class id.
        model = LibreVLM("falcon-perception", names=["building", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.xyxy)
    - label: Segment
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("falcon-perception", names=["the titanium building"])
        result = model.set_task("segment").predict(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)          # one mask per instance
    - label: 300m
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # The smaller size has no mask head: detect only.
        model = LibreVLM("falcon-perception-300m", names=["building"])
        result = model.predict(SAMPLE_IMAGE)
---

## Install

Falcon Perception belongs to LibreYOLO's VLM tier and needs the `vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download on first use from TII's own repositories,
`tiiuae/Falcon-Perception` (600m) and `tiiuae/Falcon-Perception-300M`, pinned
to a revision. The network is vendored in LibreYOLO; no remote code runs.

<code-tabs name="predict" />

The family loads through `LibreVLM()`, or the `LibreFalconPerception` class.
`names=` (or `set_classes()`) is required. Each vocabulary entry is one text
query and one generation, so cost grows with the number of entries, and every
instance found for an entry gets that entry's class id. A query can be a class
name or a phrase such as `"the dog on the left"`.

For each instance the model emits a box centre, a box extent and a mask query.
`detect` returns `Results.boxes`; `segment` (600m only) also returns
`Results.masks`, made by dotting the mask query with AnyUp-upsampled image
features. Confidence is a constant 1.0 and there is no NMS: upstream's
coordinate de-duplication runs inside decoding (`coord_dedup_threshold`,
default 0.01). The generation budget is 2048 new tokens per query
(`max_new_tokens`).

## Variants

| Name | Size | Tasks |
| --- | --- | --- |
| `falcon-perception`, `falcon-perception-600m` | 600m (default) | detect, segment |
| `falcon-perception-300m` | 300m | detect |

Both run at 1024 px. When only MPS is available the family runs on CPU,
because its RoPE tables are complex64, which MPS does not implement. On CPU in
fp32 the port matches the upstream engine bit for bit: logits and hidden
states on every forward pass, and the generated tokens, box floats and mask
bytes.

## Export

Inference only: `train()`, `val()`, `export()` and `chat()` raise.

## Licensing

<provenance-box>

Falcon Perception is Apache-2.0, code and weights, by the
[Technology Innovation Institute](https://github.com/tiiuae/Falcon-Perception).
Commercial use is allowed; TII's NOTICE asks public-facing work to retain it
and cite the project. The AnyUp feature upsampler, used only by the 600m
segment task, is by the AnyUp authors under CC BY 4.0, which permits
commercial use with attribution. LibreYOLO does not redistribute the weights.

</provenance-box>

## Citation

<citation-block />
