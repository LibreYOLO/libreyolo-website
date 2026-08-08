---
title: SwinIR
families: [swinir]
seo_title: "SwinIR: run 4x image super-resolution in LibreYOLO"
description: "Use SwinIR in LibreYOLO for 4x image super-resolution. Install, predict, validate and export the lightweight, medium and large checkpoints."
lead: "A Swin Transformer network for image restoration. LibreYOLO ships inference and validation for its 4x super-resolution checkpoints: the official lightweight, real-world medium and real-world large generators."
keywords: [SwinIR, Swin Transformer, image super-resolution, image restoration, residual Swin Transformer block]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSwinIRm-restore.pt source=bus.jpg save=True
    - label: Tiled, for large images
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile splits the forward pass into overlapping tiles and blends the
        # seams back together; tile_pad is the halo added around each tile
        # before it is cropped back out. Both are Python-only keyword
        # arguments, not CLI flags.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # imgsz defaults to a small internal patch size when omitted, not
        # your working resolution, so pass the size your deployment actually
        # feeds the model.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
---

## Install

SwinIR needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A restore result carries no boxes; `result.restored` is a dense `(H, W, 3)`
uint8 RGB image, on a canvas 4x the input in each dimension. `save=True` writes
that image directly rather than an annotated plot. The input is padded to a
multiple of 8 rather than resized, so predict runs at the photo's own
resolution; a source larger than memory allows can be split with `tile` and
`tile_pad`, which blend the tile seams back together in the output. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Three sizes, all fixed at a 4x upscale. `s` is the official lightweight
generator, with four residual Swin Transformer block (RSTB) stages and
pixel-shuffle-direct upsampling. `m` and `l` are the real-world medium and
large generators, with six and nine RSTB stages and a nearest-neighbor-plus-
convolution upsampler built for real-world degradations rather than only
bicubic downscaling.

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
`Results`. ExecuTorch and every format the matrix marks blocked are not
available for this family; ONNX, TorchScript, TensorRT, OpenVINO and TFLite
are. [Export](/docs/export) lists the arguments every format accepts and the
extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
