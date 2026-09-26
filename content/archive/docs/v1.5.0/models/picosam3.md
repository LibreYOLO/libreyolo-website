---
title: PicoSAM3
families: [picosam3]
seo_title: "PicoSAM3: box-prompted edge segmentation in LibreYOLO"
description: "Use PicoSAM3 in LibreYOLO for box-prompted region segmentation on edge sensors. Install, predict and export the pico checkpoint under Apache-2.0."
lead: "PicoSAM3 is a compact CNN distilled from SAM 2.1 and SAM 3, built for box-prompted region-of-interest segmentation on sensors like the Sony IMX500. LibreYOLO supports it through a dedicated LibreSAM factory, separate from the LibreYOLO() detector factory, with box prompts only."
keywords: [PicoSAM3, Segment Anything, edge segmentation, region of interest, box prompt, in-sensor inference, IMX500, knowledge distillation]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Box prompt
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # PicoSAM3 has a single size, "pico", so no other alias is needed.
        model = LibreSAM("picosam3")

        # bboxes= is the only supported prompt: [x1, y1, x2, y2] or a list of
        # boxes, one mask per box. Each box is expanded 10%, made square,
        # clipped to the image and resized to 96x96 before the CNN runs.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # polygon per mask
        print(result.boxes.xyxy)    # tight box derived from the mask
    - label: Encode once, prompt many
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image() caches the source image; PicoSAM3 runs one full CNN
        # forward per box, so this saves the image load/decode, not an
        # encoder pass the way it does for the other SAM families.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (default 13) and dynamic (default True, batch axis only) are
        # the only export arguments this family accepts.
    - label: Use the exported file
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # PicoSAM3 exports its raw 96x96 ROI CNN: roi_image -> mask_logits.
        # There is no LibreYOLO-side pre/postprocessing to reuse here, since
        # export() is not routed back through LibreYOLO() the way a detector
        # checkpoint is.
        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")
        name = session.get_inputs()[0].name
        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96), dtype=np.float32)})

        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
---

## Install

PicoSAM3 needs the `sam` extra: LibreYOLO's own weight download still goes
through `transformers`' Hugging Face tooling, even though inference runs on a
native, non-`transformers` CNN.

```bash
pip install "libreyolo[sam]"
```

## Predict

`LibreSAM(...)` (or the family-specific `LibrePicoSAM3(...)`) is a separate
entry point from `LibreYOLO(...)`: it returns a promptable segmenter rather
than a detector, because a forward pass here is meaningless without a
prompt. There is no `libreyolo predict` CLI command for this family; use the
Python API.

<code-tabs name="predict" />

PicoSAM3 accepts only `bboxes=`; passing `points=`, `labels=`, `masks=`,
`text=`, `multimask=True` or omitting the box to segment everything all
raise a clear `ValueError`, since none of those modes exist in the upstream
model. `conf` filters by predicted mask quality (IoU), not a detection
confidence, and must be between `0.0` and `1.0`. Every mask carries class id
`0`, named `"object"`. `train()`, `val()` and `track()` raise
`NotImplementedError`; use LibreSAM2 or LibreSAM3 for point, text, mask or
segment-everything prompts. See [prediction](/docs/predict) for source types.

## Variants

One size, pico, at a fixed 96 px ROI input: PicoSAM3 runs one full CNN
forward per box rather than encoding the whole image once.

## Export

<export-matrix />

PicoSAM3 is the only family in the SAM tier that exports: it ships its raw
96x96 ROI CNN to ONNX, `roi_image -> mask_logits`, with no NMS or mask
post-processing baked in. The other SAM families raise `NotImplementedError`
on `export()`, since their encoder/decoder split has no defined runtime export
contract yet. An exported PicoSAM3 graph does not load back through
`LibreYOLO()`; run it directly with a runtime such as `onnxruntime`, applying
the same 10%-padded square-ROI preprocessing shown above.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

PicoSAM3 is distilled from SAM 2.1 and SAM 3 as teacher models. LibreYOLO
does not vendor or redistribute either teacher's code or weights in this
family; only the compact student CNN and its converted checkpoint are
shipped.

</provenance-box>

## Citation

<citation-block />
