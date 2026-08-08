---
title: Real-ESRGAN
families: [realesrgan]
seo_title: "Real-ESRGAN: image super-resolution in LibreYOLO"
description: "Use Real-ESRGAN in LibreYOLO for practical image super-resolution at 4x, 2x and a fast 4x tier. Install, predict, validate and export."
lead: "A practical blind super-resolution upscaler trained on synthetic degradations rather than only bicubic downscaling. LibreYOLO ships inference and validation for its 4x, 2x and fast 4x checkpoints."
keywords: [Real-ESRGAN, RRDBNet, SRVGGNetCompact, image super-resolution, image restoration, blind super-resolution]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        results = model(SAMPLE_IMAGE, save=True)

        restored = results[0].restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRealESRGANx4-restore.pt source=bus.jpg save=True
    - label: Tiled, for large images
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile splits the forward pass into overlapping tiles and blends the
        # seams back together; tile_pad is the halo added around each tile
        # before it is cropped back out. Both are Python-only keyword
        # arguments, not CLI flags.
        results = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRealESRGANx4-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # imgsz defaults to a small internal patch size when omitted, not
        # your working resolution, so pass the size your deployment actually
        # feeds the model.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx imgsz=512
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].restored.array.shape)
---

## Install

Real-ESRGAN needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A restore result carries no boxes; `results[0].restored` is a dense `(H, W, 3)`
uint8 RGB image, on a canvas `Results.restore_scale` times the input in each
dimension. `save=True` writes that image directly rather than an annotated
plot. Input is converted to RGB and any alpha channel is dropped. A source
larger than memory allows can be split with `tile` and `tile_pad`, which blend
the tile seams back together in the output. See [prediction](/docs/predict)
for sources, streaming and result handling.

## Variants

Three checkpoints, named for their upscale factor. `x4` is RRDBNet
(`RealESRGAN_x4plus`), 23 residual-in-residual dense blocks, the quality
default at 4x. `x2` is the same RRDBNet architecture at 2x. `x4t` is
SRVGGNetCompact (`realesr-general-x4v3`), a smaller, faster generator built for
video and lower-latency use at 4x. The upstream general-purpose model also
ships a paired denoise-strength network blended in at inference time; that
strength knob is not part of this port, which runs the base `x4t` generator.

## Validate

`val()` measures PSNR and SSIM between the restored output and a clean target
image, both computed in RGB on the original canvas with no border crop and no
resizing. SSIM uses an 11x11 Gaussian window with sigma 1.5, averaged over the
three color channels.

<code-tabs name="val" />

The dataset argument is a YAML pairing a directory of degraded input images
with a directory of clean target images of matching resolution; see
[dataset formats](/docs/reference/dataset-formats) for the exact keys.

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. [Export](/docs/export) lists the arguments every format accepts and
the extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
