---
title: Dome-DETR
families: [domedetr]
seo_title: "Dome-DETR: tiny-object detection in LibreYOLO"
description: "Use Dome-DETR in LibreYOLO for tiny-object detection on aerial and drone imagery. Convert the upstream weights, predict, fine-tune and validate under MIT-licensed code."
lead: "A tiny-object specialist built on D-FINE: a density head decides where objects are, encoder attention is restricted to the windows that hold them, and the query count is sized from that density instead of being fixed. LibreYOLO supports it for detection."
keywords: [Dome-DETR, tiny object detection, small object detection, aerial imagery, drone detection, remote sensing, VisDrone, AI-TOD, DETR, density adaptive queries]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Convert, then predict
      language: bash
      code: |
        # No Dome-DETR weights are hosted by LibreYOLO, so the checkpoint is
        # fetched from the upstream repository and converted once.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A local path, not a bare name: nothing downloads for this family.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        result = model("drone-frame.jpg", save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt source=drone-frame.jpg save=True
    - label: Class names
      language: python
      code: |
        from libreyolo import LibreYOLO

        # There is no COCO checkpoint, so the classes come from the dataset the
        # weights were trained on and are read from the checkpoint metadata.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 AI-TOD-V2 classes

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 VisDrone classes
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4, lr0=2e-4)
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

There is nothing to auto-download. LibreYOLO does not host these weights, so
the flow is: fetch the upstream checkpoint, convert it once, then load the
converted file by path. [Licensing](#licensing) explains why.

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

LibreYOLO publishes no benchmark rows for this family, because it publishes no
checkpoints to benchmark.

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

There are none to list. LibreYOLO publishes no Dome-DETR weights, and no name
of the form `LibreDOMEDETR<size>-<dataset>.pt` resolves to a download.

Upstream publishes six checkpoints, s, m and l for each of two datasets:
AI-TOD-V2 with 9 classes and VisDrone with 12. There is no COCO checkpoint, so
a canonical filename always carries the dataset suffix, and the class names
travel in the checkpoint metadata rather than coming from a family constant.
Asking for a bare `LibreDOMEDETRs.pt` raises immediately with a message naming
the two real filenames and the conversion command, rather than attempting a
download that would 404.

`weights/convert_domedetr_weights.py` does the conversion. It rebuilds the
LibreYOLO graph, loads the upstream tensors into it, and refuses to write
anything if a single key is missing, unexpected or the wrong shape, so a
converted file is either an exact match or it does not exist. Point it at an
upstream `.pth` and pass the size and the variant:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

On numerical fidelity, `weights/parity_domedetr.py` compares this port against
the upstream implementation across all six checkpoints and reports
`max_abs_diff == 0.0` on both `pred_logits` and `pred_boxes`, after first
checking the MWAS window mask bit for bit, and separately diffs every loss
term against upstream's criterion. Be clear about what that is: a manual script
that needs the upstream checkout and the published checkpoints on disk, run by
hand. It is not part of continuous integration, and no CI job reproduces it.

## Licensing

<provenance-box>

The weights are the reason this family is not mirrored. The upstream model card
carries no license field in its metadata, and its prose states that the project
is Apache-2.0 while also restricting the material to academic research
purposes only. Those two readings do not agree, and the stricter one is not a
redistribution grant, so LibreYOLO links the upstream repository instead of
copying the files, pending clarification. The same reasoning is what governs
[YOLO-NAS](/docs/models/yolo-nas) here.

The code is a separate question and a clearer one. The upstream repository is
Apache-2.0, LibreYOLO's port is MIT, and weights you train yourself on your own
data are yours.

</provenance-box>

## Citation

Dome-DETR was published at ACM Multimedia 2025 as "Dome-DETR: DETR with
Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection". The preprint is at
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). The authors
publish no BibTeX block in their repository, so none is reproduced here rather
than assembled by hand.

<citation-block />
