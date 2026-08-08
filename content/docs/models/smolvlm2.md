---
title: SmolVLM2
families: [smolvlm2]
seo_title: "SmolVLM2 in LibreYOLO: open-vocabulary detection"
description: "SmolVLM2 in LibreYOLO: install, set an open vocabulary and predict or chat with Hugging Face's Apache-2.0 vision-language model."
lead: "SmolVLM2 is Hugging Face's small vision-language model. LibreYOLO wraps it as an open-vocabulary object detector and exposes its free-form chat directly: supply a class list to detect, or ask it a question."
keywords: [SmolVLM2, vision-language model, open-vocabulary detection, small multimodal model, Hugging Face, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        results = model.predict(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # The escape hatch beneath the detection convenience: any question,
        # not just a bounding-box query.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
---

## Install

SmolVLM2 belongs to LibreYOLO's VLM-as-detector tier, a separate product
surface from the checkpoint-based families with its own factory. It needs the
`vlm` extra, which also pulls in `num2words`, a dependency of SmolVLM2's own
processor.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

This family loads through the `LibreVLM()` factory, not `LibreYOLO()`: VLM
families declare no checkpoint loader, so the file-suffix routing described on
other model pages does not apply here. `set_classes()` sets the vocabulary
SmolVLM2 is asked to find; it is sticky, so it stays in effect across every
later `predict()`/`track()` call until you set it again. SmolVLM2 needs no
parser override in LibreYOLO: it follows the same chat-template-plus-JSON
output as the tier's shared default, so its detection prompt and box format
are not family-specific. Every detection carries the same placeholder
confidence, so `conf` filtering is all-or-nothing rather than a ranking; `iou`
does have an effect, dropping a later same-class box once it overlaps an
already-kept one past the threshold, since a repeating generator can otherwise
emit near-duplicate boxes for one object. SmolVLM2 also answers free-form
questions through `chat()`, the same escape hatch documented on the `LibreVLM`
factory. LibreYOLO's CLI does not cover this tier: there is no
`libreyolo predict model=...` form for it. See [prediction](/docs/predict) for
sources, streaming and result handling.

## Variants

One size in the registry: SmolVLM2-500M-Video-Instruct, loaded as
`LibreVLM("smolvlm2-500m")`. SmolVLM2 is a weaker detector than the
purpose-built grounding models in this tier; LibreYOLO's own wrapper describes
it as a demonstration that a new family needs no special-case parsing to work
here, not as its strongest open-vocabulary option.

LibreYOLO does not train, validate or export SmolVLM2: `train()`, `val()` and
`export()` all raise `NotImplementedError` for every family in this tier (see
the support tier above). Fine-tune SmolVLM2 upstream and load the resulting
weights if you need a custom vocabulary baked in; check `predict()` output by
eye instead of a COCO-style validation pass, since every detection carries the
same placeholder confidence.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
