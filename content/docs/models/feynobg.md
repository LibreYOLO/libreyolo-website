---
title: FeyNobg
families: [feynobg]
seo_title: "FeyNobg: background removal in LibreYOLO"
description: "Use FeyNobg in LibreYOLO for background removal and alpha matting, a deepened BiRefNet variant from Feyn Inc. Install, predict and validate."
lead: "A background-removal model from Feyn Inc. that deepens BiRefNet's architecture and retrains it. LibreYOLO ships inference and validation for FeyNobg's matte task."
keywords: [FeyNobg, background removal, dichotomous image segmentation, alpha matte, image matting, cutout, nobg]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFeyNobgl-matte.pt source=bus.jpg save=True
    - label: Cutout
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: source RGB plus the matte as an alpha channel.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # A directory containing images/ and an auto-detected matte directory
        # (mattes/, matte/, gt/, masks/, mask/ or alpha/) also works in place
        # of a dataset YAML.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
---

## Install

FeyNobg needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

The checkpoint downloads from the LibreYOLO organization on Hugging Face on
first use and is cached locally, the same as any other family, though it is
not yet listed in the Checkpoints table on this page.

<code-tabs name="predict" />

A matte result carries no boxes; `result.matte` is a dense `(H, W)`
float32 array in `[0, 1]`, 1 fully foreground and 0 fully background. Unlike a
binary mask, the soft matte keeps anti-aliased edge detail such as hair and
fur. `result.cutout()` composites the source image with that alpha channel
into an RGBA array, and `result.save(path)` (or `save=True` on the predict
call) writes it straight to a transparent-background PNG. The model runs at a
fixed native 1024x1024 canvas; a different resolution is not supported,
because the Swin backbone's relative-position tables are tied to it, and a
mismatch interpolates them badly rather than raising an error. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

One published size, `l`, a Swin-L tier backbone. FeyNobg takes BiRefNet's
architecture and deepens its third Swin stage from 18 to 24 blocks before
retraining, so the LibreYOLO port reuses BiRefNet's forward path,
preprocessing and single-logit output contract; predict, validate and
checkpoint handling behave the same as the `birefnet` family.

## Validate

`val()` reports two metrics over a paired image/matte folder, both in
`[0, 1]` and independent of resolution: MAE, the mean absolute error against
the ground-truth alpha (lower is better), and S-measure (Fan et al., ICCV
2017), a structural similarity that credits preserving the subject's shape and
holes, which pixel MAE alone misses (higher is better). Validation drives the
model's own `predict`, so it uses the family's exact preprocessing.

<code-tabs name="val" />

Validation is inference-only. The upstream `nobg` library ships Apache-2.0
training code; fine-tuning today means training there and converting the
result with LibreYOLO's own conversion script, not calling `train()` on this
family, which raises rather than running a partial trainer.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
