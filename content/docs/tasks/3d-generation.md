---
title: 3D generation
seo_title: "Image to 3D in Python: 3D Gaussian splats with LibreYOLO"
description: "Turn one picture of an object into the whole object as 3D Gaussian splats in LibreYOLO: the LibreGen call, seeds, the payload, .ply and .splat files."
lead: "3D generation turns one picture of an object into the whole object, including the sides the picture never showed. LibreYOLO exposes it as the generate3d task, served by the LibreGen factory, and returns the object as 3D Gaussian splats."
keywords: [image to 3D python, single image to 3D, 3D Gaussian splatting, Gaussian splats, 3DGS, PLY, splat, Speridlabs Idol, LibreGen, libreyolo generate3d task]
last_verified: "1.7.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        # An example image from the Speridlabs Idol SDK. A local path, a PIL
        # image, a NumPy array or a folder of images works the same way.
        IMAGE = "https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp"

        model = LibreGen("speridlabs-idol")   # weights download on first use
        result = model(IMAGE, seed=0)

        result.save("armchair.ply")           # standard 3DGS layout
        result.save("armchair.splat")         # compact web format

        g = result.gaussians
        print(len(g), g.xyz.shape, g.colors.shape)   # N, (N, 3), (N, 3) RGB
        print(g.camera, g.seed)                      # the inferred input view, the seed
    - label: CLI
      language: bash
      code: |
        # Saves the object as .ply under runs/generate/predict.
        libreyolo generate source=https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/demo/examples/nb-armchair.webp seed=0
---

## Definition

The `generate3d` task returns one synthesized 3D object per result, in
`result.gaussians` as a `Gaussians` payload. The object sits in its own frame,
z up and centered in the cube `[-0.5, 0.5]^3`. Each row is one Gaussian:
`xyz` positions `(N, 3)`, `scales` linear standard deviations `(N, 3)`,
`rotations` unit quaternions `(N, 4)` in `(w, x, y, z)` order, `opacities`
`(N,)` in `[0, 1]`, and `sh` spherical-harmonic color coefficients
`(N, K, 3)`, with `colors` the `(N, 3)` RGB that the degree-0 term encodes.
`camera` is the input view the model inferred, `seed` and `params` reproduce
the object, and `input_image` is the RGB image the model saw after its own
background removal and crop. The full payload is described in
[Results types](/docs/reference/results-types#gaussians).

`result.boxes` is `None`; `result.path`, `result.orig_shape` and
`result.orig_img` describe the source image, and `len(result)` is 1.
`result.summary()` returns one row with the payload's record.

The canonical task key is `generate3d`. `generation3d`, `3d-generation`,
`image-to-3d`, `image23d`, `img2-3d`, `img-to-3d` and `i23d` normalize to it.

## Models

[Speridlabs Idol](/docs/models/speridlabs-idol), by Speridlabs, is the only
family serving this task. It is ported from the Apache-2.0 Speridlabs SDK and
runs through `LibreGen`; `LibreYOLO()` does not load it. Its settings, speed
and memory figures are on the model page.

## Predict

<code-tabs name="predict" />

`model(...)`, `model.predict(...)` and `model.generate(...)` are the same
call. The source is anything predict takes: a path, URL, `PIL.Image`, NumPy
array (BGR), tensor, a folder of images, or a list of them. An image file or
`PIL.Image` with real transparency is used as the object's cutout; any other
input has its background removed first. The input resolution is fixed by the
model, so `imgsz` raises, and so does `prompt=`: an image-to-3D model has no
text input.

One source with `n=1` returns a `Results`; several sources or `n > 1` return a
flat list in source order, and `stream=True` a generator. A seed, with the
same model and settings, gives the same object, and object `i` of a source
uses `seed + i`. Without a seed, a fresh one is drawn and recorded in
`result.gaussians.seed`.

`save()` picks the format from the extension. A `.ply` is the standard 3D
Gaussian Splatting layout, turned y-up the way splat viewers and trainers
expect, with the seed, camera and settings stored in a header comment. A
`.splat` is the compact web format, 32 bytes per Gaussian. `save=True` writes
each object as a `.ply` to a new `runs/generate/predict` folder per call,
the file named after its index, source and seed; `output_path` names one `.ply` or `.splat` file
instead, for one source with `n=1`. `result.plot()` returns a point render from
the inferred input view, a quick look rather than a Gaussian rasterization.
`g[g.opacities > 0.1]` selects Gaussians and returns a new `Gaussians`.

The [`libreyolo generate`](/docs/cli/generate) command runs the same call from
the shell with `source=` and saves by default.

## Train

No family in this task trains inside LibreYOLO: `train()` raises
`NotImplementedError`, and `train(data=...)` is reserved for fine-tuning on
pairs of images and 3D objects.

## Validate

There is no 3D generation metric yet: `val()` raises `NotImplementedError`,
and `val(data=...)` is reserved for rendered-view metrics on a held-out set.

## Export

Export is not implemented. A sampling loop over several networks has no
exportable graph contract yet, so `export()` raises `NotImplementedError`.
