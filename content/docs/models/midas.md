---
title: MiDaS
families: [midas]
seo_title: "MiDaS: monocular depth estimation in LibreYOLO"
description: "Use MiDaS in LibreYOLO for monocular depth estimation. Install, predict, validate and export two MIT-licensed variants, downloaded from isl-org."
lead: "MiDaS is monocular relative depth estimation trained with scale-and-shift invariant loss across mixed datasets, the line of work that established the zero-shot depth transfer protocol later families reuse. LibreYOLO supports it for the depth task: predict and zero-shot validation, with no training path."
keywords: [MiDaS, monocular depth estimation, DPT, relative depth, depth map, zero-shot depth]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Not on disk yet: LibreYOLO downloads it from the official isl-org/MiDaS
        # GitHub release and checks it against a pinned SHA-256 before use.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        results = model(SAMPLE_IMAGE, save=True)

        depth = results[0].depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMiDaSl-depth.pt source=bus.jpg save=True
    - label: Small variant
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # EfficientNet-Lite3 encoder, smaller and faster than the DPT-Large l size.
        model = LibreYOLO("LibreMiDaSs-depth.pt")
        results = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].depth_map.data.shape)
---

## Install

MiDaS needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

MiDaS is the one depth family LibreYOLO does not republish on its own Hugging
Face organization. Requesting a checkpoint by its LibreYOLO filename downloads
the matching official asset directly from the `isl-org/MiDaS` GitHub releases,
checks it against a pinned SHA-256, and wraps it with LibreYOLO's checkpoint
metadata before first use; later runs reuse the cached local file. See
Licensing for why.

<code-tabs name="predict" />

`results[0].depth_map` carries a dense relative inverse-depth map: higher
values mean closer to the camera, and the values have no metric unit or
cross-image scale. `save=True` writes a colormapped visualization of that map
to disk; `Results.plot()` does not cover this family, since it is defined for
surface normals and edges only. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Two variants with different encoders, not just different scales of the same
one. `s` is MiDaS v2.1 Small, an EfficientNet-Lite3 encoder. `l` is DPT-Large,
a ViT-L/16 encoder with the DPT decoder MiDaS introduced for dense prediction.
They also preprocess differently: `s` uses an upper-bound aspect resize with
ImageNet mean/std normalization, `l` uses a minimal aspect resize with mean and
std of 0.5. Pick `s` for a lighter CNN, `l` for the transformer decoder's
accuracy.

Training is not offered for this family. `LibreMiDaS.train()` raises
`NotImplementedError` unconditionally.

## Validate

`val()` runs the shared depth validator: it aligns each prediction to its
ground truth with a per-image least-squares scale and shift, then reports the
standard zero-shot relative-depth metrics, AbsRel, RMSE and the three delta
thresholds.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`, with `depth_map` in place of boxes.

<code-tabs name="export" />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
