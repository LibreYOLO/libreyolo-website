---
title: Image restoration
seo_title: "Image restoration and upscaling in LibreYOLO"
description: "Denoise, deblur and upscale images in LibreYOLO. Predict a restored RGB image, train NAFNet on paired data, and read the PSNR and SSIM keys."
lead: "Image restoration takes a degraded image and returns a clean one. LibreYOLO exposes it as the restore task, which covers denoising, deblurring and super-resolution behind a single output contract: one RGB image in, one RGB image out."
keywords: [image restoration python, image denoising model, image super resolution python, deblurring model, PSNR SSIM validation]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Upscale an image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The compact 4x generator; tile bounds peak memory on a large source.
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        results = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        results[0].restored.save("upscaled.png")
        print(results[0].restored.array.shape)   # 4x the input in each axis
    - label: Denoise an image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Trained on SIDD real-image noise; output stays at the input size.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        results = model(SAMPLE_IMAGE)

        results[0].restored.save("denoised.png")
        print(results[0].restore_scale)   # 1: no upscale for this checkpoint
  train:
    - label: Fine-tune NAFNet on paired images
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16, lr0=1e-3)
    - label: Record the provenance on the checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation and dataset are written into the saved checkpoint for
        # provenance; they take no part in training.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() returns a plain dict, not an object.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz is fixed into the graph, so pass the size your deployment
        # actually feeds the model.
        model.export(format="onnx", imgsz=256)
    - label: Run the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        results = model(SAMPLE_IMAGE)

        results[0].restored.save("denoised.png")
---

## Definition

The `restore` task maps one image to another image. Denoising, deblurring and
super-resolution are all the same task here, because they share one contract:
the model consumes an RGB image and returns an RGB image, and the degradation it
was trained to undo is a property of the checkpoint rather than of the API.

A prediction fills `results[0].restored`, a `RestoredImage` payload holding an
`(H, W, 3)` uint8 RGB array. `.array` returns it as NumPy and `.save(path)`
writes it to disk. `results[0].restore_scale` records the upscale factor the
output canvas carries, which is `1` for a checkpoint that preserves resolution.
`results[0].boxes` stays empty, so `conf`, `iou` and `max_det` are accepted for
signature parity but have no effect, and `save=True` writes the restored image
directly rather than an annotated photo.

## Models

Three families serve `restore`, split by the degradation they undo.

[NAFNet](/docs/models/nafnet) is the denoiser, and the only restore family
LibreYOLO can train. Its architecture replaces the nonlinear activations of a
UNet block with elementwise multiplication, and the published checkpoint is
trained on SIDD real-image noise. Output stays at the input resolution.

[Real-ESRGAN](/docs/models/real-esrgan) is the practical upscaler: three
checkpoints trained against synthetic degradations rather than only bicubic
downscaling, at 4x, 2x, and a smaller, faster 4x generator built for lower
latency.

[SwinIR](/docs/models/swinir) upscales 4x with a Swin Transformer backbone, in
three sizes covering the official lightweight generator and two real-world
generators.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

Restoration runs at the source image's own resolution rather than a fixed
network canvas, padding only to the network's downsample factor, so both time
and memory scale with the pixel count of your input. `tile` splits the forward
pass into overlapping tiles and blends the seams back together, and `tile_pad`
is the halo added around each tile before it is cropped back out; both are
Python keyword arguments. See [prediction](/docs/predict) for sources, streaming
and result handling.

## Dataset format

Restoration pairs each degraded input image with a clean target image of exactly
the same resolution, matched by filename stem.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` and `names` are schema placeholders; a restore model returns
`Results.restored`, not detections. `degradation` and `dataset` are optional
provenance labels. `target_stem_suffix` covers datasets that name the clean
image differently from its degraded pair. Validation keeps native resolution and
pads only enough to stack a batch, so the metrics are computed on the original
canvas. See [dataset formats](/docs/reference/dataset-formats) for the full
contract.

## Train

NAFNet is the only restore family with a training implementation.
`Real-ESRGAN.train()` and `SwinIR.train()` both raise `NotImplementedError`:
those checkpoints come from GAN training over synthetic degradation pipelines,
and the paired restore trainer would run without reproducing that recipe.

<code-tabs name="train" />

The trainer takes coupled crops of the input and target pair, so both sides stay
aligned. See [training](/docs/train) for datasets, multi-GPU and loggers, and
the [NAFNet page](/docs/models/nafnet) for this family's defaults and the
inference-time pooling it detaches while training.

## Validate

`val()` compares the restored output against the clean target, in RGB, on the
original canvas, with no border crop and no resizing.

<code-tabs name="val" />

`metrics/PSNR` is the peak signal-to-noise ratio in decibels, and it is also
`fitness`, the number best-checkpoint selection reads. `metrics/SSIM` is
structural similarity in `[0, 1]`, computed with an 11x11 Gaussian window at
sigma 1.5 and averaged over the three color channels. Higher is better for both.

## Export

An exported restore model loads back through `LibreYOLO()` on its file suffix,
so a `.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`, with `restored` carrying the output image.

<code-tabs name="export" />

Restore export fixes the spatial resolution into the graph, so pass the `imgsz`
your deployment will actually feed the model. For NAFNet that size must divide
by the network's downsample factor, and only the batch dimension stays dynamic
under `dynamic=True`. For Real-ESRGAN and SwinIR, leaving `imgsz` out falls back
to a small internal patch size rather than your working resolution. Per-format
coverage is on each model page and in the
[full export matrix](/docs/reference/export-matrix). [Export](/docs/export)
lists the arguments every format accepts.
