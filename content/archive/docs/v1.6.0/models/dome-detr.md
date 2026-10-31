---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: tiny-object detection in LibreYOLO'
description: >-
  Use Dome-DETR for tiny-object detection, training and validation. Mirrored
  pretrained checkpoints retain academic-research-only terms.
lead: >-
  A tiny-object specialist built on D-FINE: a density head decides where objects
  are, encoder attention is restricted to the windows that hold them, and the
  query count is sized from that density instead of being fixed. LibreYOLO
  supports it for detection.
keywords:
  - Dome-DETR
  - tiny object detection
  - small object detection
  - aerial imagery
  - drone detection
  - remote sensing
  - VisDrone
  - AI-TOD
  - DETR
  - density adaptive queries
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Pretrained weights are restricted to academic research.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
---

## Install

Dome-DETR needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

The six converted checkpoints download automatically from LibreYOLO mirrors. Their upstream terms restrict use to academic research.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the query
selection; `iou` is accepted for API parity but has no effect, because the
decoder is a set predictor with no NMS step. See [prediction](/docs/predict)
for sources, streaming and result handling.

Two capabilities are off for this family. CUDA graph capture is disabled,
because PAQI's query count is data dependent and the forward pass therefore
changes shape from image to image, which is exactly what graph capture cannot
absorb. Test-time augmentation runs at a single fixed square size, so a
multi-scale TTA request is a no-op.

## Variants

Three sizes, s, m and l, all at 800 by 800. The size selects the backbone, and
the dataset the weights came from selects the decoder depth and the query
budget, so a size code on its own does not identify a graph. AI-TOD-V2 weights
select between 300 and 1500 queries per image, VisDrone weights between 250 and
500, and the large model runs four decoder layers on AI-TOD-V2 against six on
VisDrone.

Dome-DETR is D-FINE with three additions. DeFE predicts a density map. MWAS
uses that map to restrict encoder attention to the windows that actually hold
objects, rather than attending everywhere. PAQI sizes the query set from the
same density instead of decoding a fixed 300. The gain concentrates where
objects are smallest, and narrows as they grow: upstream's own ablation moves
AP on very tiny objects from 14.0 to 17.8 while AP on medium objects moves only
from 45.4 to 46.4. Treat it as a companion to
[D-FINE](/docs/models/d-fine) for aerial, drone and remote-sensing imagery, not
a replacement for it.

No Vision Analysis benchmark rows are recorded for this family.

## Train

Dome-DETR is trainable. Training runs upstream's full objective: the D-FINE
losses plus DeFE density and count supervision, with padded queries masked out
of the classification terms and per-image denoising attention masks so one
image's padding cannot leak into another's.

<code-tabs name="train" />

The configuration inherits D-FINE's recipe and changes what MWAS requires.
`imgsz` is 800, `lr0` is `2e-4`, the backbone parameter group is scaled by
`backbone_lr_mult=0.1`, and `multi_scale` is forced off, because MWAS windows
need the input to stay divisible by stride 8. `batch` defaults to 4 rather than
D-FINE's 16: PAQI pads every batch to its widest member, so memory tracks the
busiest image in the batch rather than the average one.

One honest caveat about accuracy. Upstream trains for 160 epochs on
`MultiStepLR(milestones=[80, 120], gamma=0.8)`, while these defaults run
D-FINE's flat-cosine schedule for the same 160 epochs. That schedule has not
been reproduced here and the paper's AP numbers have not been reproduced
either, so read them as the upstream authors' results rather than as a promise
that this recipe reaches them. Supply the upstream schedule if matching the
paper is the goal.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary keyed by metric name, and prints per-class results
when `verbose` is left on.

<code-tabs name="val" />

Validation runs against your own dataset in the format you trained on. The
library's COCO validation gate does not apply here, since no COCO checkpoint
exists for this family to be measured against.

## Export

Export is not supported, for every format, and asking for one raises rather
than producing a file.

The reason is PAQI. It decides the query count per image, from
density-filtered proposals and a greedy density-adaptive suppression loop, so
the decoder's output length is a property of the input rather than of the
graph. Tracing bakes in whichever count the tracing image happened to produce,
which yields an artifact that silently returns wrong results for every other
image. A static formulation would have to unroll that suppression over all 250
to 1500 candidates, and collapsing to a fixed top-k would remove exactly the
tiny-object recall the family exists for. If you need an exportable detection
transformer, [D-FINE](/docs/models/d-fine) is the one to reach for.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box>

The six mirrors retain the upstream academic-research-only restriction. The code is a separate licensing surface. The upstream repository is
Apache-2.0, LibreYOLO's port is MIT, and weights you train yourself on your own
data are yours.

</provenance-box>
